# Render .docx / .pptx files to PDF with the locally installed Microsoft Office (for visual QA).
# Usage: powershell -ExecutionPolicy Bypass -File tools/render_office.ps1 <file> [<file> ...]
# The PDF is written next to each input file (same name, .pdf extension).
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Files)

$ErrorActionPreference = "Stop"
$word = $null
$ppt = $null
try {
    foreach ($f in $Files) {
        $full = (Resolve-Path $f).Path
        $pdf = [System.IO.Path]::ChangeExtension($full, ".pdf")
        $ext = [System.IO.Path]::GetExtension($full).ToLower()
        if ($ext -eq ".docx") {
            if ($null -eq $word) { $word = New-Object -ComObject Word.Application; $word.Visible = $false; $word.DisplayAlerts = 0 }
            $doc = $word.Documents.Open($full, $false, $true)
            # 17 = wdFormatPDF. Show tracked changes as markup in the PDF.
            $doc.ExportAsFixedFormat($pdf, 17, $false, 0, 0, 0, 0, 7)
            $doc.Close($false)
        } elseif ($ext -eq ".pptx") {
            if ($null -eq $ppt) { $ppt = New-Object -ComObject PowerPoint.Application }
            $pres = $ppt.Presentations.Open($full, $true, $false, $false)
            # 32 = ppSaveAsPDF
            $pres.SaveAs($pdf, 32)
            $pres.Close()
        } else {
            Write-Output "skip $full"
            continue
        }
        Write-Output "rendered $pdf"
    }
} finally {
    if ($null -ne $word) { $word.Quit() | Out-Null; [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null }
    if ($null -ne $ppt) { $ppt.Quit() | Out-Null; [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null }
}
