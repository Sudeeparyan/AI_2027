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
            # PowerPoint's actual text bounds catch clipping that estimating
            # line counts in the JavaScript builder cannot reliably predict.
            $layoutIssues = @()
            foreach ($slide in $pres.Slides) {
                foreach ($shape in $slide.Shapes) {
                    # PowerPoint grows table rows to fit their text, so a table
                    # whose real bottom passes the content area has overflowed.
                    if ($shape.HasTable -eq -1 -and $shape.Top + $shape.Height -gt 7.0 * 72) {
                        $layoutIssues += [pscustomobject]@{
                            slide = $slide.SlideIndex; shape = $shape.Name; text = "table"
                            height = $shape.Height; width = $shape.Width; top = $shape.Top; left = $shape.Left
                        }
                    }
                    if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
                        $frame = $shape.TextFrame2
                        $text = $frame.TextRange
                        $right = $text.BoundLeft + $text.BoundWidth
                        $bottom = $text.BoundTop + $text.BoundHeight
                        # Ink bounds include font bearings (even a right-aligned
                        # page number overhangs its frame by about 2.25 points).
                        # Four points avoids those false alarms, while detecting
                        # an extra wrapped line. Verify flagged boxes visually.
                        $tolerance = 4
                        if ($text.BoundLeft -lt $shape.Left - $tolerance -or $right -gt $shape.Left + $shape.Width + $tolerance -or
                            $text.BoundTop -lt $shape.Top - $tolerance -or $bottom -gt $shape.Top + $shape.Height + $tolerance) {
                            $layoutIssues += [pscustomobject]@{
                                slide = $slide.SlideIndex; shape = $shape.Name
                                text = $text.Text; height = $shape.Height; width = $shape.Width
                                boundHeight = $text.BoundHeight; boundWidth = $text.BoundWidth
                                boundLeft = $text.BoundLeft; boundTop = $text.BoundTop
                                left = $shape.Left; top = $shape.Top
                            }
                        }
                    }
                }
            }
            ConvertTo-Json -InputObject @($layoutIssues) -Depth 4 | Set-Content -LiteralPath ([System.IO.Path]::ChangeExtension($full, '.layout.json')) -Encoding UTF8
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
