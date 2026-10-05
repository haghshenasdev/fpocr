$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$dir = Join-Path $root 'assets\models'
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$files = @(
  @{
    Name='PP-OCRv5_mobile_det.onnx'
    Url='https://huggingface.co/vladadu/pp-ocrv5-arabic-mobile-onnx/resolve/main/PP-OCRv5_mobile_det.onnx?download=true'
    Sha256='c8d9b07063420ce5365c74e42532de48238feeeedcdb7a330b195708bc38a93d' # source SHA256
  },
  @{
    Name='arabic_PP-OCRv5_mobile_rec.onnx'
    Url='https://huggingface.co/vladadu/pp-ocrv5-arabic-mobile-onnx/resolve/main/arabic_PP-OCRv5_mobile_rec.onnx?download=true'
    Sha256='4e2f4ae42104e1b272463966c56ddafa3c6ad98ce9d8c7ed765ce66666ea13e1'
  },
  @{
    Name='ppocrv5_arabic_dict.txt'
    Url='https://huggingface.co/vladadu/pp-ocrv5-arabic-mobile-onnx/resolve/main/arabic_PP-OCRv5_mobile_rec_dict.txt?download=true'
    Sha256='7f92f7dbb9b75a4787a83bfb4f6d14a8ab515525130c9d40a9036f61cf6999e9'
  }
)

foreach ($f in $files) {
  $target = Join-Path $dir $f.Name
  Write-Host "Downloading $($f.Name)..."
  Invoke-WebRequest -Uri $f.Url -OutFile $target
  $hash = (Get-FileHash -Algorithm SHA256 $target).Hash.ToLowerInvariant()
  Write-Host "SHA256: $hash"
  if ($hash -ne $f.Sha256) {
    Write-Warning "SHA256 differs from the recorded source hash. Do not use a partial/LFS pointer file."
  }
}

Write-Host ''
Write-Host 'Models are in assets\models. Run: flutter pub get'
