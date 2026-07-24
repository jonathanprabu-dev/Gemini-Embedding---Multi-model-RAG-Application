@echo off
REM Starts the multimodal RAG app and opens it in Chrome.
REM Runs at logon via a shortcut in shell:startup (see README "Start on boot").
REM Safe to run by hand at any time: it no-ops on anything already running.
REM
REM Deliberately batch, not PowerShell: a PowerShell version that polled the
REM port in a retry loop was blocked outright by Defender's AMSI heuristics.

setlocal
set "PROJECT=C:\dev\Gemini Embedding Restruchered"
set "PYTHON=C:\Users\jonat\AppData\Local\Programs\Python\Python314\python.exe"
set "CHROME=C:\Program Files\Google\Chrome\Application\chrome.exe"
set "URL=http://localhost:8501"
set "LOG=%PROJECT%\scripts\start-rag.log"

cd /d "%PROJECT%"
echo [%date% %time%] --- logon start --->>"%LOG%"

REM ---- Ollama (local reasoning model) ----
REM Its installer normally registers its own autostart; this is a fallback so
REM the first query doesn't fail if it isn't up yet.
curl.exe -s -o nul --max-time 2 http://localhost:11434
if not errorlevel 1 goto ollama_ok
where ollama >nul 2>&1
if errorlevel 1 (
    echo [%time%] ollama not on PATH - reasoning will fail until it runs>>"%LOG%"
) else (
    start "ollama" /min ollama serve
    echo [%time%] started ollama serve>>"%LOG%"
)
:ollama_ok

REM ---- Streamlit ----
curl.exe -s -o nul --max-time 2 "%URL%"
if not errorlevel 1 (
    echo [%time%] streamlit already up>>"%LOG%"
    goto ready
)
REM --server.headless stops Streamlit opening the default browser itself; this
REM script owns opening Chrome so the browser choice is explicit.
start "rag" /min "%PYTHON%" -m streamlit run app.py --server.port 8501 --server.headless true
echo [%time%] launched streamlit>>"%LOG%"

REM Wait for the server to actually answer before opening a tab, otherwise
REM Chrome lands on connection-refused and needs a manual reload.
set /a tries=0
:wait
curl.exe -s -o nul --max-time 2 "%URL%"
if not errorlevel 1 goto ready
set /a tries+=1
if %tries% geq 45 (
    echo [%time%] TIMEOUT waiting for %URL% - not opening Chrome>>"%LOG%"
    exit /b 1
)
ping -n 3 127.0.0.1 >nul
goto wait

:ready
start "" "%CHROME%" "%URL%"
echo [%time%] opened %URL% in Chrome>>"%LOG%"
exit /b 0
