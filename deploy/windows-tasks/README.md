# Windows scheduled tasks

Exported from Task Scheduler on 25/10/2026. These tasks are not stored in the project
folder by Windows itself, so a reinstalled PC needs them created again.

| Task | When | What |
|---|---|---|
| `MeliaWildlifeTrackerWatchdog` | every 15 min | starts the tracker if it is not running, restarts it if frozen (`tools/tracker_watchdog.py`) |
| `MeliaWildlifeTrainingRound4` | 25/10/2026 20:30 (once) | night training round 4 (`training/round4.py`) |
| `MeliaGitAutoSync` | every 30 min | commits and pushes changes (detections, learned state, models, code) to GitHub (`tools/git_autosync.py`, log `logs/autosync.log`) |

Both run `tools/run_hidden.vbs` (no window) for the user who is logged on.

Install on this or another PC (Command Prompt, as the user who runs the tracker):

    schtasks /Create /TN MeliaWildlifeTrackerWatchdog /XML deploy\windows-tasks\MeliaWildlifeTrackerWatchdog.xml /F

The XML holds this PC's user SID (`<UserId>`) and the path
`C:\Users\Melia Serengeti User\Desktop\wildlife_tracker`.  On another PC or user, change
both first: put the new user name in `<UserId>` and the project folder in the paths.
For a later training round, change the date in `<StartBoundary>` and the `.bat` name.
