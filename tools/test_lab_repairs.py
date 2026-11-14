"""Exercise repaired source contracts without data downloads, keys or API calls."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def source_tree(week):
    raw = (ROOT / f"curriculum/labs/week_{week:02d}_lab.py").read_text(encoding="utf-8")
    return ast.parse("\n".join(line for line in raw.splitlines() if not line.startswith("%")))


class LabRepairTests(unittest.TestCase):
    def test_entire_hosted_prompt_excludes_local_dataset_examples(self):
        nodes = [n for n in source_tree(8).body if
                 (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HOSTED_EXAMPLES" for t in n.targets))
                 or (isinstance(n, ast.FunctionDef) and n.name == "hosted_prompt")]
        self.assertEqual(len(nodes), 2)
        ns = {"INTENTS": ["card_arrival", "lost_or_stolen_card", "exchange_rate", "top_up_failed", "age_limit"],
              "INSTRUCTION": "Classify the invented customer message.",
              "SHOTS": [{"text": "DATASET_SENTINEL", "label_text": "card_arrival"}],
              "train_ds": [{"text": "DATASET_SENTINEL", "label_text": "card_arrival"}]}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "hosted_source", "exec"), ns)
        prompt = ns["hosted_prompt"]("INVENTED_QUERY")
        self.assertNotIn("DATASET_SENTINEL", prompt)
        self.assertIn("INVENTED_QUERY", prompt)
        for example, label in ns["HOSTED_EXAMPLES"]:
            self.assertIn(example, prompt)
            self.assertIn(label, prompt)

    def test_trainer_obeys_cpu_and_gpu_opt_in(self):
        node = next(n for n in source_tree(8).body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "args" for t in n.targets))
        for device, expected in (("cpu", True), ("cuda", False)):
            ns = {"SFTConfig": lambda **kwargs: SimpleNamespace(**kwargs), "DEVICE": device,
                  "SMOKE": False, "SEED": 42, "steps_per_epoch": 5}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "trainer_source", "exec"), ns)
            self.assertIs(ns["args"].use_cpu, expected)
            self.assertFalse(ns["args"].fp16)
            self.assertFalse(ns["args"].bf16)

    def test_validation_is_explicit_and_final_test_is_guarded(self):
        tree = source_tree(8)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "evaluate"]
        self.assertEqual(len(calls), 6)
        dev = [n for n in calls if isinstance(n.args[2], ast.Name) and n.args[2].id == "dev_ds"]
        test = [n for n in calls if isinstance(n.args[2], ast.Name) and n.args[2].id == "test_ds"]
        self.assertEqual(len(dev), 3)
        self.assertEqual(len(test), 3)
        guard = next(n for n in tree.body if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Constant)
                     and n.test.left.value == "_W08_FINAL_TEST")
        guarded = [n for n in ast.walk(ast.Module(body=guard.orelse, type_ignores=[])) if n in test]
        self.assertEqual(len(guarded), 3, "Every final-test evaluation must be inside the once-only branch")

    def test_portable_completion_loss_matches_reference_and_adapter_gradients(self):
        try:
            import torch
            from peft import LoraConfig, get_peft_model
            from transformers import Qwen2Config, Qwen2ForCausalLM
            from transformers.modeling_outputs import CausalLMOutput
            from trl import SFTTrainer
        except ImportError:
            self.skipTest("Run this numerical regression in the pinned lab environment")

        node = next(n for n in source_tree(8).body if isinstance(n, ast.ClassDef)
                    and n.name == "PortableSFTTrainer")
        ns = {"torch": torch, "SFTTrainer": SFTTrainer, "CausalLMOutput": CausalLMOutput}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "portable_sft_source", "exec"), ns)
        torch.manual_seed(42)
        causal = Qwen2ForCausalLM(Qwen2Config(vocab_size=23, hidden_size=16, intermediate_size=32,
                                           num_hidden_layers=1, num_attention_heads=4,
                                           num_key_value_heads=2, max_position_embeddings=16))
        model = get_peft_model(causal, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                                 target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        model.eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.02)
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]]),
                  "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
                  "labels": torch.tensor([[-100, -100, -100, 4, 5], [-100, -100, 8, 9, -100]])}
        fake_trainer = SimpleNamespace(accelerator=SimpleNamespace(unwrap_model=lambda value: value))
        compute = ns["PortableSFTTrainer"].compute_loss
        loss, outputs = compute(fake_trainer, model, inputs, return_outputs=True)
        self.assertIs(outputs.loss, loss)
        self.assertEqual(tuple(outputs.logits.shape), (4, 23), "Only non-masked next-token positions are projected")
        reference = model(**inputs, use_cache=False).loss
        torch.testing.assert_close(loss, reference, atol=1e-6, rtol=1e-6)
        loss.backward()
        portable_grads = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                          if parameter.requires_grad}
        self.assertTrue(any(float(gradient.abs().sum()) > 0 for gradient in portable_grads.values()),
                        "Real adapter gradients must reach trainable LoRA matrices")
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad))
        model.zero_grad(set_to_none=True)
        reference.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, portable_grads[name], atol=1e-6, rtol=1e-5)
        normalised = compute(fake_trainer, model, inputs, num_items_in_batch=8)
        torch.testing.assert_close(normalised, reference.detach() / 2, atol=1e-6, rtol=1e-6)
        empty = {**inputs, "labels": torch.full_like(inputs["labels"], -100)}
        empty_loss = compute(fake_trainer, model, empty)
        self.assertEqual(float(empty_loss), 0.0)
        self.assertTrue(empty_loss.requires_grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
