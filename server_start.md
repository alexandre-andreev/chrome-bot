I'll scan the repository to find a `requirements` file or other dependency manifest and confirm where `exa_py` is imported, so I can give you exact install commands for PowerShell.

[5 tools called]

I found `backend/requirements.txt` and the import in `backend/app/main.py`. I'll open both to confirm whether `exa_py` is declared and guide the exact install command for your PowerShell venv.

[2 tools called]

��宦�, � �����쭮� venv �� ��⠭������ ����ᨬ���. � `backend/requirements.txt` ���� ����� `exa-py`, � ������ � ���� - `from exa_py import Exa`, ⠪ � ������ ����. � �।� Cursor ����ᨬ��� 㦥 ����, ���⮬� ⠬ ��� ����᪠����.

������� � PowerShell (� ��୥ �஥��), ����� ��⨢�஢������ venv:

```powershell
# ������� pip � ���⠢�� ����ᨬ��� �஥��
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt

# �����⨬ �ࢥ� �१ �����, �⮡� �筮 �ᯮ�짮����� venv
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

�᫨ �訡�� ��࠭����, ����� �஢���� ��⠭����:
```powershell
python -c "import exa_py, sys; print('exa_py:', exa_py.__file__, '\npython:', sys.executable)"
python -m pip show exa-py
python -m pip freeze | Select-String exa
```
�᫨ `exa-py` �� ������ - ��⠭���� �:
```powershell
python -m pip install exa-py
```

����� 㡥�����, �� �� ����᪠�� ������� �� ��⨢�஢������ venv (� �ਣ��襭�� ������ ���� `(.venv)`), � �� ⥪��� ��४��� - ��७� ९������.

- ������: ���� `GEMINI_API_KEY` � `EXA_API_KEY` ����� �������� � `.env.local` � ��୥ �஥�� (��� 㦥 ��� ���墠�뢠��), �� �� ������⢨� �� ��뢠�� `ModuleNotFoundError`.

- � ������� ����� �� `python -m uvicorn ...`, �⮡� �᪫���� ��⠭��� � �ਯ⠬� ��� venv.

- �᫨ �� �� Python 3.13 � ���� �����-� ����� �� �����ঠ�, ᮮ��� �� �訡�� ��⠭���� - ���᪠�� ���� ���ᨩ.