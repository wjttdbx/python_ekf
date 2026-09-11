param(
    [string]$Path = "C:\Users\何逸凡\OneDrive\桌面\专利-EKF\100002说明书_双测量模式_估计距离判据_原生公式_20260727.docx"
)

$ErrorActionPreference = "Stop"
$resolved = (Resolve-Path -LiteralPath $Path).Path
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0

function Convert-AllMatchesToMath {
    param(
        [object]$Document,
        [string]$VisibleText,
        [string]$UnicodeMath
    )

    $matches = [System.Collections.Generic.List[object]]::new()
    $search = $Document.Content.Duplicate
    $search.Find.ClearFormatting()
    $search.Find.Text = $VisibleText
    $search.Find.Forward = $true
    $search.Find.Wrap = 0
    while ($search.Find.Execute()) {
        $matches.Add([pscustomobject]@{
            Start = $search.Start
            End = $search.End
        })
        $next = $search.End
        $search.SetRange($next, $Document.Content.End)
        $search.Find.ClearFormatting()
        $search.Find.Text = $VisibleText
        $search.Find.Forward = $true
        $search.Find.Wrap = 0
    }

    foreach ($match in ($matches | Sort-Object Start -Descending)) {
        $range = $Document.Range($match.Start, $match.End)
        $range.Text = $UnicodeMath
        $start = $range.Start
        $mathRange = $Document.Range($range.Start, $range.End)
        $null = $Document.OMaths.Add($mathRange)
        $math = $Document.Range($start, $mathRange.End).OMaths.Item(1)
        if ($null -eq $math) {
            throw "OMath creation failed for: $VisibleText"
        }
        $math.BuildUp()
        $math.Type = 1
    }
}

try {
    $document = $word.Documents.Open($resolved, $false, $false)

    $replacements = @(
        @("δx0=[0.036846,-0.125755,0.090745,0.00013133,-0.00027242,-0.00018182]", "δx_0=[0.036846,-0.125755,0.090745,0.00013133,-0.00027242,-0.00018182]"),
        @("x̂0=x0+δx0", "x̂_0=x_0+δx_0"),
        @("Reff=R/(1-γ⁻²)=2×10¹³I3", "R_(eff)=\frac(R)(1-γ^(-2))=2×10^13 I_3"),
        @("Reff⁻¹=(1-γ⁻²)R⁻¹=0.5×10⁻¹³I3", "R_(eff)^(-1)=(1-γ^(-2))R^(-1)=0.5×10^(-13) I_3"),
        @("S=BReff⁻¹Bᵀ", "S=B R_(eff)^(-1) B^T"),
        @("AᵀP+PA-PBReff⁻¹BᵀP+Qsdre=0", "A^T P+PA-PB R_(eff)^(-1) B^T P+Q_(sdre)=0"),
        @("u=-Reff⁻¹BᵀP x̂", "u=-R_(eff)^(-1) B^T P x̂"),
        @("Rz=diag(0.01²,(1.4×10⁻⁴)²,(1.4×10⁻⁴)²)", "R_z=diag(0.01^2,(1.4×10^(-4))^2,(1.4×10^(-4))^2)"),
        @("R=10¹³I3", "R=10^13 I_3"),
        @("Qsdre=I6", "Q_(sdre)=I_6"),
        @("P_EKF,0", "P_(EKF,0)"),
        @("ν0=0", "ν_0=0"),
        @("γ=√2", "γ=\sqrt(2)"),
        @("d̂<dmin", "d̂<d_(min)"),
        @("dmin=100 m", "d_(min)=100 m"),
        @("d̂<100 m", "d̂<100 m"),
        @("δx0", "δx_0")
    )

    foreach ($replacement in $replacements) {
        Convert-AllMatchesToMath -Document $document `
            -VisibleText $replacement[0] -UnicodeMath $replacement[1]
    }

    $document.Save()
    $document.Close($false)
}
finally {
    if ($null -ne $document) {
        try { $document.Close($false) } catch {}
    }
    $word.Quit()
}
