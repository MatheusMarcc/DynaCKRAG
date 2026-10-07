<#
  ragbench — roda os benchmarks do RAG-Critic e do DynaKRAG, de ponta a ponta.

  Uso (PowerShell, na pasta do projeto):
      .\run_all.ps1 -Painel                 # abre o PAINEL no navegador (rodar, ver falhas, explorar os dados)
      .\run_all.ps1                         # perfil quick (padrão), os dois artigos, tudo pela linha de comando
      .\run_all.ps1 -Perfil mini            # checagem rápida (30 perguntas por dataset) antes da rodada longa
      .\run_all.ps1 -Perfil smoke           # teste de 1 minuto, sem Ollama (LLM simulado)
      .\run_all.ps1 -Paper dynakrag         # só um artigo (ragcritic | dynakrag)
      .\run_all.ps1 -Tier 1                 # só os resultados principais (tier 1): o que o painel precisa
      .\run_all.ps1 -Workers 2              # 2 perguntas em paralelo (exige OLLAMA_NUM_PARALLEL=2, ver docs/USO.md)

  O painel também abre com dois cliques em painel.bat.

  Se a execução parar (queda de energia, Ctrl+C, suspensão), rode o MESMO comando de novo:
  tudo o que já foi feito está em cache e a rodada continua de onde parou.

  Se o PowerShell bloquear o script:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#>
param(
    [ValidateSet("smoke", "mini", "quick", "standard")] [string]$Perfil = "quick",
    [ValidateSet("all", "dynakrag", "ragcritic")] [string]$Paper = "all",
    [ValidateSet("all", "1", "2", "3")] [string]$Tier = "all",
    [int]$Workers = 1,
    [switch]$Painel,
    [int]$Porta = 8765,
    [switch]$SkipInstall,
    [switch]$SkipPull,
    [switch]$NoKeepAwake
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:TOKENIZERS_PARALLELISM = "false"
New-Item -ItemType Directory -Force -Path "ragbench_results\logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Transcript -Path "ragbench_results\logs\run_all_${Perfil}_$stamp.txt" -Append | Out-Null

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

try {
    # ------------------------------------------------------------------ Python + venv
    Step "Python"
    function Test-Py([string]$cmd) {
        # testa sem deixar o PowerShell transformar o stderr do py.exe em erro fatal (ruído no log)
        $old = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $out = & ([scriptblock]::Create("$cmd -c `"import sys; print(sys.version_info[:2] >= (3,10))`"")) 2>$null
            return ($LASTEXITCODE -eq 0 -and "$out".Trim() -eq "True")
        } catch { return $false } finally { $ErrorActionPreference = $old }
    }
    if (Test-Path ".venv\Scripts\python.exe") {
        # já existe um .venv: usa ele (não procura outros Pythons)
        if (-not (Test-Py "& '.venv\Scripts\python.exe'")) {
            throw "O .venv desta pasta tem Python < 3.10 ou está quebrado. Apague a pasta .venv e rode de novo."
        }
    } else {
        $cands = @()
        if (Get-Command py -ErrorAction SilentlyContinue) { $cands += @("py -3.12", "py -3.11", "py -3.10", "py -3") }
        if (Get-Command python -ErrorAction SilentlyContinue) { $cands += "python" }
        $py = $null
        foreach ($cand in $cands) { if (Test-Py $cand) { $py = $cand; break } }
        if (-not $py) { throw "Python 3.10+ não encontrado. Instale em https://www.python.org/downloads/ (marque 'Add to PATH')." }
        Write-Host "criando .venv com $py"
        & ([scriptblock]::Create("$py -m venv .venv"))
        if (-not (Test-Path ".venv\Scripts\python.exe")) { throw "não consegui criar o .venv com $py" }
    }
    $vpy = (Resolve-Path ".venv\Scripts\python.exe").Path
    & $vpy --version
    $doInstall = -not $SkipInstall
    if ($Painel -and $doInstall) {
        # para abrir o painel depressa: só instala se faltar alguma coisa
        $old = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $vpy -c "import yaml, httpx, numpy, scipy, sklearn, tqdm, json_repair, matplotlib, datasets, sentence_transformers" 2>$null
        if ($LASTEXITCODE -eq 0) { $doInstall = $false }
        $ErrorActionPreference = $old
    }
    if ($doInstall) {
        Step "Dependências (primeira vez: alguns minutos; o torch CPU tem ~200 MB)"
        & $vpy -m pip install --upgrade pip --quiet
        & $vpy -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw "pip install falhou" }
    }

    # ------------------------------------------------------------------ Ollama
    # No modo -Painel, um problema com o Ollama (não instalado, não responde, sem internet para conferir os modelos)
    # vira aviso: o painel abre do mesmo jeito para ver os resultados; só não dá para rodar testes novos.
    $ollamaOk = $true
    if ($Perfil -ne "smoke") {
    Step "Ollama"
    try {
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        throw "Ollama não encontrado. Instale em https://ollama.com/download e rode este script de novo."
    }
    $up = $false
    try { Invoke-RestMethod -Uri "http://localhost:11434/api/version" -TimeoutSec 5 | Out-Null; $up = $true } catch {}
    if (-not $up) {
        Write-Host "iniciando 'ollama serve' em segundo plano..."
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        for ($i = 0; $i -lt 30 -and -not $up; $i++) {
            Start-Sleep -Seconds 2
            try { Invoke-RestMethod -Uri "http://localhost:11434/api/version" -TimeoutSec 5 | Out-Null; $up = $true } catch {}
        }
        if (-not $up) { throw "O Ollama não respondeu em http://localhost:11434. Abra o app do Ollama e tente de novo." }
    }
    } catch {
        if (-not $Painel) { throw }
        $ollamaOk = $false
        Write-Warning "$($_.Exception.Message)"
        Write-Warning "O painel abre mesmo assim, para ver os resultados. Para rodar testes novos o Ollama precisa estar funcionando."
    }
    }
    if (-not $SkipPull -and $Perfil -ne "smoke" -and $ollamaOk) {
        $models = @("qwen2.5:3b", "hf.co/mradermacher/RAG-Critic-3B-GGUF:Q4_K_M")
        if ($Perfil -eq "standard") { $models += "llama3.2:3b" }
        foreach ($m in $models) {
            Write-Host "ollama pull $m"
            & ollama pull $m
            if ($LASTEXITCODE -ne 0) {
                if (-not $Painel) { throw "falha ao baixar $m" }
                Write-Warning "não consegui baixar/conferir $m (sem internet?). O painel abre mesmo assim."
            }
        }
        # opcional: o mesmo BGE-large do artigo servido pelo Ollama, para montar os índices na GPU (bem mais rápido
        # que na CPU). Se não baixar, o índice é feito na CPU — só demora mais.
        Write-Host "ollama pull bge-large  (embeddings na GPU para montar os índices; opcional)"
        & ollama pull bge-large
        if ($LASTEXITCODE -ne 0) { Write-Warning "não baixei o bge-large: os índices serão montados na CPU (mais devagar, mesmo resultado)." }
    }

    # ------------------------------------------------------------------ não deixar o Windows dormir
    if (-not $NoKeepAwake) {
        if (-not ("RagBench.Power" -as [type])) {
            Add-Type -Namespace RagBench -Name Power -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
"@
        }
        # ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x1): impede a suspensão enquanto o script roda
        [RagBench.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null
        Write-Host "suspensão automática bloqueada enquanto o benchmark roda (mantenha a máquina na tomada)."
    }

    $common = @("--profile", $Perfil)
    if ($Workers -gt 1) { $common += @("--set", "llm.workers=$Workers") }

    if ($Painel) {
        Step "painel (Ctrl+C nesta janela para fechar)"
        $uiArgs = @("--port", $Porta)
        # sem -Perfil explícito, o painel abre no perfil que tem resultado mais recente (dá para trocar na página)
        if ($PSBoundParameters.ContainsKey("Perfil")) { $uiArgs += @("--profile", $Perfil) }
        if ($Workers -gt 1) { $uiArgs += @("--set", "llm.workers=$Workers") }
        & $vpy -m ragbench ui @uiArgs
        return
    }

    Step "doctor (modelos, % na GPU, estimativa de tempo)"
    & $vpy -m ragbench doctor @common
    if ($LASTEXITCODE -ne 0 -and $Perfil -ne "smoke") { Write-Warning "o doctor apontou problemas (veja acima). Continuando mesmo assim..." }

    Step "prepare (datasets, corpora, índices, RAG-Error Bench)"
    & $vpy -m ragbench prepare @common
    if ($LASTEXITCODE -ne 0) { throw "prepare falhou — veja o log em ragbench_results\logs" }

    Step "bench ($Paper) — tier 1 → 2 → 3"
    $tierArgs = @()
    if ($Tier -ne "all") { $tierArgs = @("--tier", $Tier) }
    & $vpy -m ragbench bench @common --paper $Paper @tierArgs
    if ($LASTEXITCODE -ne 0) { throw "bench parou com erro — rode o mesmo comando de novo para continuar" }

    Step "relatórios"
    & $vpy -m ragbench report @common
    Write-Host "`nPronto. Para ver as falhas de cada método e abrir cada teste:  .\run_all.ps1 -Painel   (ou dois cliques em painel.bat)" -ForegroundColor Green
    Write-Host "Relatórios em ragbench_results\$Perfil\: REPORT.md (índice), FALHAS.md (análise de falhas), COMPARACAO.md (comparação com os artigos)." -ForegroundColor Green
    Write-Host "Planilhas: metrics.csv, comparacao.csv, falhas_por_teste.csv, falhas_resumo.csv. Log detalhado: ragbench_results\logs\" -ForegroundColor Green
}
finally {
    if (-not $NoKeepAwake) {
        try { [RagBench.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null } catch {}
    }
    Stop-Transcript | Out-Null
}
