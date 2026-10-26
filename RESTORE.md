# Kurudisha mfumo kwenye PC mpya / iliyosafishwa

Repository: https://github.com/proches/wildlife-tracking-system (private)

Mfumo ulifanya kazi kwenye: Windows 10 Pro, Intel i5-7200U (CPU tu), **Python 3.13.15**,
kamera SUNBA ONVIF PTZ kwenye `10.140.0.12`.

## 1. Programu za kusakinisha kwanza
1. **Python 3.13 (64-bit)** kutoka https://www.python.org/downloads/ . Usitumie 3.14 — SpeciesNet haiikubali bado.
   Wakati wa kusakinisha, weka tiki "py launcher".
2. **Git for Windows** kutoka https://git-scm.com/download/win .

## 2. Pakua project
Fungua Command Prompt:

    cd /d "%USERPROFILE%\Desktop"
    git -c credential.helper=manager clone https://github.com/proches/wildlife-tracking-system.git wildlife_tracker
    cd wildlife_tracker

Repository ni **private**: dirisha la GitHub litafunguka uingie kwa akaunti `proches`
(`credential.helper=manager` ndiyo inayofungua dirisha hilo; bila hiyo git inasema
"could not read Username").

Weka project kwenye `C:\Users\<jina>\Desktop\wildlife_tracker`.  Mtumiaji akiwa tofauti na
"Melia Serengeti User", badilisha njia (path) ndani ya `training\run_round3.bat`,
`training\run_round4.bat` na XML za `deploy\windows-tasks\`.

## 3. Sakinisha Python packages (matoleo yaleyale)
Bonyeza mara mbili **`setup.bat`**.  Inaunda `.venv313`, inasakinisha `requirements-lock.txt`
(matoleo kamili ya mfumo uliokuwa unafanya kazi, pamoja na torch ya CPU) na inaendesha majaribio.

## 4. Siri (`.env`) — HAZIPO GitHub
`setup.bat` inaunda `.env` kutoka `.env.example`.  Fungua `.env` na uweke:

    WILDLIFE_CAMERA_PASSWORD=<password ya kamera>

Password ya kamera haihifadhiwi GitHub kwa makusudi — itunze mahali salama (password manager).
Mipangilio mingine iliyokuwa kwenye `.env` ya zamani: IP `10.140.0.12`, mtumiaji `admin`, RTSP `tcp`,
`WILDLIFE_CUSTOM_YOLO_ENABLED=true`, `WILDLIFE_CUSTOM_YOLO_MODEL_PATH=models/melia_wildlife.pt`,
YouTube imezimwa.

## 5. Model na data kutoka "Releases"
Model za detection ziko ndani ya repository (`models\melia_wildlife.pt`, `models\MDV6-yolov9-c.pt`).
Vingine viko kwenye ukurasa wa **Releases** wa repository (release `data-2026-10-25`):

| Faili | Ni nini | Ifungue wapi |
|---|---|---|
| `speciesnet-model.zip` | SpeciesNet (spishi 2000+), MB 306 | `%USERPROFILE%\.cache\` (inaunda `kagglehub\...`). Bila hii, mfumo unaipakua wenyewe ukiwa na internet mara ya kwanza. |
| `site-data-1.zip`, `site-data-2.zip` | picha za kamera hii zilizowekewa labels (training), GB 3.5 | folda ya project (zinaunda `training_data\site*`, `missed`) |
| `lila-data.zip` | Snapshot Serengeti iliyokwisha kupakuliwa, GB 1.5 | folda ya project |
| `lila4-data.zip` | Snapshot Serengeti ya round 4 (swala, nyumbu, ndege ...), picha 3 289, MB 600 | folda ya project (`training_data\lila4_yolo`) |
| `round4-run.zip` | log za round 4 na matokeo ya kila epoch (`results.csv`, grafu) | folda ya project, si lazima |
| `review-frames-1.zip`, `review-frames-2.zip` | picha za review za mfumo (chanzo cha training zijazo), GB 2 | folda ya project (`logs\review\`) |
| `logs-history.zip` | historia ya logs/traces (ushahidi wa uchunguzi) | folda ya project (`logs\`), si lazima |

Video za highlights (`logs\highlights`, GB 10) hazikupakiwa.

Data ya training inahitajika tu ukitaka kufanya training tena; tracker inafanya kazi bila hiyo.

Training inayopakua picha zaidi za Snapshot Serengeti (`training\round4.py`) inahitaji pia metadata
ya LILA (GB 5.2, haipo GitHub kwa sababu ya ukubwa).  Ipakue kutoka
https://lila.science/datasets/snapshot-serengeti na uweke faili hizi mbili ndani ya `training_data\lila\`:
`SnapshotSerengetiBboxes_20190903.json` na `SnapshotSerengeti_S1-11_v2.1.json`.
Folda za `training_data\dataset*` na `site3s` hazihifadhiwi kwa sababu scripts za training zinazijenga upya.

## 6. Washa
Bonyeza mara mbili **`run_tracker.bat`**.  Log: `logs\tracker.log`.  Mara ya kwanza ukiona
`Species classifier: Google SpeciesNet`, `MegaDetector ready` na `Observation patrol ACTIVE`, kila kitu kiko sawa.

## 7. Tasks za Windows (mlinzi + training za usiku)
Tazama `deploy\windows-tasks\README.md`.  Kwa ufupi (baada ya kurekebisha `<UserId>` kama mtumiaji ni mpya):

    schtasks /Create /TN MeliaWildlifeTrackerWatchdog /XML deploy\windows-tasks\MeliaWildlifeTrackerWatchdog.xml /F
    schtasks /Create /TN MeliaGitAutoSync /XML deploy\windows-tasks\MeliaGitAutoSync.xml /F

`MeliaGitAutoSync` inapakia GitHub kila dakika 30 mabadiliko yoyote (wanyama walioonekana, model, code).

## 8. Mipangilio ya Windows iliyokuwa imewekwa
- Power: kompyuta isilale (Sleep: Never) — training na tracker zinafanya kazi usiku.
- Akaunti ibaki imeingia (logged on): tasks zinaendeshwa kwa mtumiaji aliyeingia.
- QuickEdit ya cmd: tracker inaizima yenyewe (`utils/logger.py`).

## Maelezo ya mfumo
`README.md` (mfumo mzima na matukio ya uwanjani) na `docs\notes\` (tabia za kamera, ushahidi wa
detection/tracking, training za usiku, hardware).
