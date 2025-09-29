I'll scan the repository to find a `requirements` file or other dependency manifest and confirm where `exa_py` is imported, so I can give you exact install commands for PowerShell.

[5 tools called]

I found `backend/requirements.txt` and the import in `backend/app/main.py`. I'll open both to confirm whether `exa_py` is declared and guide the exact install command for your PowerShell venv.

[2 tools called]

Похоже, в локальном venv не установлены зависимости. В `backend/requirements.txt` есть пакет `exa-py`, а импорт в коде - `from exa_py import Exa`, так и должно быть. В среде Cursor зависимости уже стоят, поэтому там всё запускается.

Сделайте в PowerShell (в корне проекта), внутри активированного venv:

```powershell
# обновим pip и поставим зависимости проекта
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt

# запустим сервер через модуль, чтобы точно использовался venv
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Если ошибка сохранится, быстро проверить установку:
```powershell
python -c "import exa_py, sys; print('exa_py:', exa_py.__file__, '\npython:', sys.executable)"
python -m pip show exa-py
python -m pip freeze | Select-String exa
```
Если `exa-py` не найден - установите явно:
```powershell
python -m pip install exa-py
```

Также убедитесь, что вы запускаете команды из активированного venv (в приглашении должно быть `(.venv)`), и что текущая директория - корень репозитория.

- Важное: ключи `GEMINI_API_KEY` и `EXA_API_KEY` можно положить в `.env.local` в корне проекта (код уже его подхватывает), но их отсутствие не вызывает `ModuleNotFoundError`.

- Я изменил запуск на `python -m uvicorn ...`, чтобы исключить путаницу со скриптами вне venv.

- Если вы на Python 3.13 и вдруг какой-то пакет не поддержан, сообщите об ошибке установки - подскажу пины версий.