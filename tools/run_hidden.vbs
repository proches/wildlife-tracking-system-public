' Runs a .bat with no window and waits for it; used by Task Scheduler:
'     wscript.exe //B "...\tools\run_hidden.vbs" "...\training\run_round3.bat"
' 23/10/2026: the visible console of the night training was closed in the morning
' (exit 0xC000013A), so its last step - restarting the tracker - never ran.  A job
' without a window cannot be closed by accident; the tracker it starts still opens
' its own visible window (training/round2.py start_tracker).
Set sh = CreateObject("WScript.Shell")
rc = sh.Run("cmd.exe /c """ & WScript.Arguments(0) & """", 0, True)
WScript.Quit rc
