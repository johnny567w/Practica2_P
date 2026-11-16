param(
    [string]$Photo = 'data\img3.jpg',
    [string]$DataDir = 'data'
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$full = (Resolve-Path $Photo).Path
$bmpPath = Join-Path (Resolve-Path $DataDir).Path ([IO.Path]::GetFileNameWithoutExtension($full) + '.bmp')
$img = [System.Drawing.Image]::FromFile($full)
try {
    if ($img.PropertyIdList -contains 0x0112) {
        $rot = switch ($img.GetPropertyItem(0x0112).Value[0]) {
            2 { 'RotateNoneFlipX' } 3 { 'Rotate180FlipNone' } 4 { 'Rotate180FlipX' } 5 { 'Rotate90FlipX' }
            6 { 'Rotate90FlipNone' } 7 { 'Rotate270FlipX' } 8 { 'Rotate270FlipNone' } default { $null }
        }
        if ($rot) { $img.RotateFlip([System.Drawing.RotateFlipType]::$rot) }
    }
    Write-Output ("{0}: {1}x{2} px ({3})" -f $Photo, $img.Width, $img.Height, $img.PixelFormat)
    $img.Save($bmpPath, [System.Drawing.Imaging.ImageFormat]::Bmp)
} finally {
    $img.Dispose()
}
# 6000, 4000 y 2000: entradas de la practica. 500: resolucion menor comun para comparar las cuatro
# versiones cuando Python secuencial no alcanza a terminar las resoluciones grandes.
& .\bin\prepare_data.exe --out $DataDir --tamanos 6000,4000,2000,500 $bmpPath
$code = $LASTEXITCODE
Remove-Item -LiteralPath $bmpPath
exit $code
