# Template PowerShell config for rerunning vacuolar phenotype inference.
# Edit inputs_lif.txt, inputs_ome_dir.txt, or inputs_params.txt instead of editing this file.

$Repo = $PSScriptRoot
Set-Location -Path $Repo

$Python = "python"

$Model = Join-Path $Repo "models\Model.keras"
$ClassMap = Join-Path $Repo "models\Model_class_map.csv"
$OutDir = Join-Path $Repo "outputs"

$Pipeline = Join-Path $Repo "vacuolar_pipeline.py"
$ApplyFilter = Join-Path $Repo "apply_filter.py"

$InputsLif = Join-Path $Repo "inputs_lif.txt"
$InputsOmeDir = Join-Path $Repo "inputs_ome_dir.txt"
$InputsParams = Join-Path $Repo "inputs_params.txt"

$CropSize = 96
$Threshold = 0.06
$SigmaMin = 3
$SigmaMax = 12
$BatchSize = 128
$MinConf = 0.60
$MinMargin = 0.10
$DisableFilter = $false

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

# Ensure the class map sits next to the model as "<model>_class_map.csv"
$ExpectedClassMap = [System.IO.Path]::ChangeExtension($Model, $null) + "_class_map.csv"
if ((Test-Path $ClassMap) -and ($ClassMap -ne $ExpectedClassMap) -and (-not (Test-Path $ExpectedClassMap))) {
  Copy-Item $ClassMap $ExpectedClassMap -Force
}

if (Test-Path $InputsParams) {
  $paramMap = @{}
  Get-Content $InputsParams |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -ne "" -and -not $_.StartsWith("#") } |
    ForEach-Object {
      $pair = $_ -split "=", 2
      if ($pair.Length -eq 2) {
        $key = $pair[0].Trim().ToLowerInvariant()
        $val = $pair[1].Trim()
        $paramMap[$key] = $val
      }
    }

  if ($paramMap.ContainsKey("crop_size")) { $CropSize = [int]$paramMap["crop_size"] }
  if ($paramMap.ContainsKey("threshold")) { $Threshold = [double]$paramMap["threshold"] }
  if ($paramMap.ContainsKey("sigma_min")) { $SigmaMin = [double]$paramMap["sigma_min"] }
  if ($paramMap.ContainsKey("sigma_max")) { $SigmaMax = [double]$paramMap["sigma_max"] }
  if ($paramMap.ContainsKey("batch_size")) { $BatchSize = [int]$paramMap["batch_size"] }
  if ($paramMap.ContainsKey("min_conf")) { $MinConf = [double]$paramMap["min_conf"] }
  if ($paramMap.ContainsKey("min_margin")) { $MinMargin = [double]$paramMap["min_margin"] }
  if ($paramMap.ContainsKey("disable_filter")) {
    $flag = $paramMap["disable_filter"].ToLowerInvariant()
    $DisableFilter = @("1", "true", "yes", "y", "on") -contains $flag
  }
}

if (Test-Path $InputsLif) {
  $Files = Get-Content $InputsLif |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -ne "" -and -not $_.StartsWith("#") } |
    ForEach-Object {
      if ([System.IO.Path]::IsPathRooted($_)) { $_ } else { Join-Path $Repo $_ }
    }
} else {
  $Files = @()
  Write-Warning "Missing inputs file: $InputsLif"
}

foreach ($f in $Files) {
  if (-not (Test-Path $f)) {
    Write-Warning "Skipping missing file: $f"
    continue
  }
  $name = [System.IO.Path]::GetFileNameWithoutExtension($f)
  $outCsv = Join-Path $OutDir ("conteo_por_imagen_val_{0}_DUAL.csv" -f $name)

  $argsList = @(
    "--lif_file", $f,
    "--model", $Model,
    "--out_csv", $outCsv,
    "--pipeline_py", $Pipeline,
    "--crop_size", $CropSize, "--threshold", $Threshold, "--sigma_min", $SigmaMin, "--sigma_max", $SigmaMax,
    "--batch_size", $BatchSize
  )

  if ($DisableFilter) {
    $argsList += "--disable_filter"
  } else {
    $argsList += @("--min_conf", $MinConf, "--min_margin", $MinMargin)
  }

  & $Python $ApplyFilter @argsList
}

# OME-TIFF/TIFF example (uncomment and edit):
# $OmeDir = Get-Content $InputsOmeDir | Select-Object -First 1
# if ($OmeDir) {
#   if (-not [System.IO.Path]::IsPathRooted($OmeDir)) { $OmeDir = Join-Path $Repo $OmeDir }
#   & $Python $Pipeline apply `
#   --images_dir "$OmeDir" `
#   --model "$Model" `
#   --out_csv (Join-Path $OutDir "conteo_por_imagen.csv") `
#   --crop_size $CropSize --threshold $Threshold --sigma_min $SigmaMin --sigma_max $SigmaMax `
#   --batch_size $BatchSize
# }
