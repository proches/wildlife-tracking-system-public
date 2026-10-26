# Melia Serengeti Wildlife AI — v3.1

> **Nakala ya public** (code na models, bila picha za kamera wala logs): https://github.com/proches/wildlife-tracking-system-public - inasasishwa yenyewe na `tools/publish_public.py`.
>
> **Kurudisha mfumo kwenye PC mpya:** soma [RESTORE.md](RESTORE.md) (setup.bat, `.env`, data za Releases, tasks za Windows).

Camera ya PTZ (SUNBA, ONVIF) inayozunguka arc ya wanyamapori, inatumia AI kugundua wanyama na ndege, inafuatilia mnyama, na kuweka kipaumbele kwenye **water-hole**.

Ukiwasha `python main.py`, mstari wa kwanza wa log lazima useme `Melia Serengeti Wildlife AI v3.1`.

---

## Kuanza

**Njia rahisi:** bonyeza mara mbili `run_tracker.bat`. Inatumia `.venv313` (Python 3.13 + SpeciesNet).

```bash
.venv313\Scripts\activate
python -m unittest discover -s tests -t .     # lazima zipite zote
python tools/validate_deployment.py           # PASS
python tools/ptz_selftest.py                  # (mara moja; camera inasogea kidogo kisha inarudi)
python main.py                                # q = toka
```

* Password ya camera iko kwenye `.env` (`WILDLIFE_CAMERA_PASSWORD=...`), si kwenye code.
* `logs/ptz_capabilities.json` inahifadhi kile `ptz_selftest.py` ilichopima (alama ya pan, AbsoluteMove). Ukibadilisha camera au setting za camera, futa faili hilo na uendeshe selftest tena.

---

## v3.1 — kilichorekebishwa (vipimo kwenye camera halisi, 16/10/2026)

### 1. Camera kubaki upande mmoja / kuzoom out bila kurudi
**Chanzo:** camera imefungwa kichwa-chini na ina **EFlip = ON**. Inatekeleza amri za pan **kinyume** (amri ya kwenda kati inaipeleka ukingoni) lakini inaripoti position bila kugeuza. Patrol ilikwama ukingoni, ikaruka kituo, ikazoom out tena, na haikuwahi kufika kituo cha kuzoom in. Mbaya zaidi, AbsoluteMove ya selftest ya zamani ilikuwa ikiipeleka camera **upande wa hotel**.

Pia:
* GetStatus inaripoti pan kwa alama iliyogeuzwa **kila mara camera inaposafiri**, na ~5% ya wakati ikiwa imetulia (ukubwa wa thamani uko sahihi).
* Speed ≤ 0.10 hazisogezi camera kabisa, na 0.15–0.22 zinatambaa tu. Config ilikuwa na 0.07 na 0.22.
* Camera haikubali PanTilt speed ndani ya AbsoluteMove.

**Marekebisho** (`camera/onvif_camera.py`, `tracking/motion.py`, `config.py`):
* `pan_command_sign`: inasomwa kutoka EFlip ya camera na kuthibitishwa na `ptz_selftest.py`. Amri zote za pan zinageuzwa wakati wa kutumwa tu.
* Filter ya pose: read iliyo nje ya arc ambayo kinyume chake kiko ndani inasahihishwa. Inakubaliwa kama halisi tu ikibaki bila kubadilika sekunde 10.
* Speed za chini zimepandishwa juu ya dead-zone. AbsoluteMove haitumi PanTilt speed.
* Patrol ikianza karibu na kituo (mfano water-hole), inakwenda moja kwa moja kwenye pose yake **bila kuzoom out kwanza**.

**Imethibitishwa:** patrol halisi ya dakika 7 ilitembelea vituo 32 kutoka pan 0.2818 (kulia) hadi 0.7275 (water-hole), ikirudi water-hole kila vituo 2, bila kutoka nje ya arc.

### 2. Water-hole
`WATER_HOLE_POSITION` = pose uliyoelekeza camera leo: **pan 0.7275, zoom 0.145 (14.8x)**. Dwell ni sekunde 45, na camera inarudi water-hole kila vituo 2 (`WATER_HOLE_EVERY_N_STATIONS`). Zoom ya zamani 0.32 (~30x) ilikuwa kali mno: focus ilikuwa ngumu na mawimbi ya joto yaliharibu picha.

Kwa matokeo bora, chora eneo la maji mara moja:
```bash
python tools/calibrate_water_roi.py --here      # camera ikiwa kwenye water-hole
```
Hii inaipa detection ndani ya maji kipaumbele, kizingiti cha chini, na pass ya ziada ya full-resolution.

### 3. Focus
**Vipimo:** kwenye 14.8x, kosa la focus la 0.01 tu linafanya picha iwe laini mara 10–25. Kwenye AUTO, lens inajifocus yenyewe sekunde 2–3 baada ya zoom kusimama. Ukibadilisha MANUAL→AUTO, lens **haijifocus**. Kwa sababu hiyo, "hunt" ya zamani (nudge, kisha kurudisha AUTO) ingeweza kuiacha lens blur kabisa.

**Sasa** (`camera/focus.py`):
* AUTO wakati wa zoom; baada ya zoom inasubiri ukali utulie.
* Picha ikiwa bado laini, inatafuta kwa nafasi za **absolute focus** na kuweka nafasi bora. Ikishindwa, inarudisha lens **mahali ilipokuwa hasa**.
* **AI haipewi frame yenye blur** (`DETECT_REQUIRE_SHARP_FRAME`), kwa sababu blur ilikuwa chanzo cha "wanyama hewa".

### 4. Detection ya uongo (mnyama/gari bila kuwepo)
**Ushahidi kwenye logs zako:** `zebra 0.83` kwenye box ya tarehe ya camera (juu-kushoto) na `zebra 0.75` kwenye `Z:14.76X` (chini-kulia). Picha za review ni vigogo na vivuli. Chanzo kikubwa ni kwamba Python 3.14 haiwezi kuendesha SpeciesNet, kwa hiyo system ilitumia **AI4G**. AI4G ni classifier "closed-set": haina jibu la "si mnyama", kwa hiyo inaita kila kitu jina la mnyama, na v3 ilihesabu jina hilo kama uthibitisho.

**Sasa:**
* Majina ya AI4G yanaonyeshwa na `?` (mfano `zebra?`), lakini **hayathibitishi** mnyama tena. Uthibitisho unahitaji detections nyingi mfululizo na score halisi ya detector, au SpeciesNet.
* Static-clutter memory (vigogo vinavyorudi pose ile ile) sasa inafanya kazi vizuri kwa sababu AbsoluteMove inarudi pose ile ile kwa usahihi wa 0.0001.
* Masks za OSD, frames zilizotulia tu, na multi-frame confirmation bado ziko (v3).

---

### 5. Movement ya cinematic
Camera hii **inapuuza speed ya AbsoluteMove**: kila mara inaruka kwa ~36°/s. Kwa ContinuousMove ina viwango viwili tu: 0.15–0.222 ni **~0.6°/s** (laini, ndicho kinachotumika sasa: `PTZ_CINEMATIC_PAN_SPEED = 0.20`), na 0.224–0.30 ni ~7°/s. Camera pia inapunguza kasi ya pan ikiwa imezoom. Patrol ni **sweep ya polepole**, na AI inachambua picha wakati camera inasogea. Kila amri ya mwendo imepimwa ili camera isiweze kuvuka lengo hata kama PC imechelewa.

### 6. Water-hole na vituo kupitia preset za camera (17/10)
Camera hii **haiwezi kuripoti wala kupokea tilt juu ya -1.0**, lakini mbuga na water-hole viko juu ya hapo. AbsoluteMove ilikuwa ikirudisha camera chini (majani ya karibu) hata kama pan ilikuwa sahihi. Sasa kila kituo ni **preset ndani ya camera**: patrol inazunguka kwa pan pekee (tilt haiguswi) na kumalizia kwa preset.
- Water-hole = preset 1 (maji katikati, 21.8x). Baada ya nusu ya muda inazoom **30x** kwa ajili ya ndege wadogo.
- Sweep inafanyika kwenye **3.85x**, tilt ikiwa imeinuliwa kuelekea mbuga.

**Kubadilisha water-hole:** simamisha system, elekeza camera kwenye maji (web page ya camera), kisha:
```bash
.venv313\Scripts\python.exe tools\build_ptz_presets.py --water
```
Tool inahifadhi mtazamo huo pamoja na preset za vituo, na kuandika `ptz_presets.jpg` uikague kwa macho.

### 7. Kufuatilia wanyama, zoom ya taratibu, kasi ya AI (18/10)
- **Mnyama alipotea kati ya matokeo mawili ya AI:** `LOST_TARGET_SECONDS` ilikuwa 10 s, lakini AI ilichukua ~12 s kwa frame. Sasa mnyama anaachwa tu baada ya **matokeo 3 mapya ya AI bila yeye** (kikomo cha juu ni 40 s).
- **Mnyama aliyetambuliwa na SpeciesNet** (≥0.60) anathibitishwa mara moja hata kama box score ni 0.25 tu. Mfano: pundamilia "plains zebra 0.86" aliyekuwa na box 0.28 hakuwahi kufuatiliwa.
- **AI ilikuwa polepole (~12 s kwa frame):**
  - Stream ya camera ilikuwa **1080p 60 fps**: decode tu ilitumia zaidi ya thread nzima ya CPU. Sasa ni **15 fps** (backup ya mipangilio ya zamani iko `logs/camera_encoder_backup.json`).
  - Ukubwa wa picha wa MegaDetector haukuwa ukibadilika kabisa, kwa hiyo hata ROI ya tracking ilikuwa ikiendeshwa kwa 1280 px (~7 s). Sasa: 960 px ikiwa imezoom (≥10x), 640 px kwa ROI ya tracking.
  - Kwenye mfumo halisi sasa ni wastani wa **~4 s kwa frame**.
- **MegaDetector au SpeciesNet wakati mwingine haikupakia** kwa sababu import zao (zote zinatumia yolov5) zilikuwa zikifanyika kwa threads mbili kwa wakati mmoja. Sasa SpeciesNet inapakiwa kwanza.
- **Zoom ya taratibu:** camera inapuuza speed ya zoom (3.85x→21.8x kwa ~3 s). Patrol sasa inazoom kwa hatua ndogo: 3.85x→21.8x kwa **~6 s** na 30x→3.85x kwa **~8 s** (`PTZ_ZOOM_GLIDE_RATE`). Kwenye water-hole: preset `water_wide` (token 8), kisha zoom ya taratibu moja kwa moja kwenye maji, kisha preset 1.
- **Water-hole katikati:** preset 1 imewekwa upya ili maji yawe katikati ya picha. Ukihitaji kurekebisha tena (simamisha system kwanza):
```bash
.venv313\Scripts\python.exe tools\center_water_preset.py --shift SX SY
```
  `SX`/`SY` = pixels ambazo maji yanatakiwa kusogea (+SX = kulia, -SY = juu). Tool inapima kwa picha na kuandika `water_center_check.jpg`.
- **Usifungue live view ya camera kwenye browser** (WebPlugin) wakati system inaendesha: inatumia CPU nyingi (~1 core) ambayo AI inaihitaji.

### 8. Scanning na tracking yenye akili zaidi (19/10)
- **Mbogo aliyeonekana wakati wa sweep alipotea baada ya sekunde 16:** baada ya kumfungia, camera iligeuka ili kumweka katikati, lakini pass ya AI iliyofuata iliangalia eneo (ROI) la zamani. Sasa camera ikiwa imegeuka tangu matokeo ya mwisho, AI inaangalia **picha nzima**.
- **Kufidia mwendo wa camera:** box za wanyama zinahamishwa kulingana na jinsi camera ilivyogeuka kati ya matokeo mawili (`PAN_PIXELS_PER_UNIT_AT_1X`, kimepimwa kwa mbogo huyo). Hivyo mnyama anabaki kuwa "yuleyule" wakati wa sweep, na anathibitishwa haraka.
- **Simama-uhakiki:** sweep ikiona kitu cha kutiliwa shaka (score ≥0.30), camera inasimama kwa sekunde 6–14 kukihakiki, kisha inaendelea. Haisimami mara mbili mahali palepale ndani ya dakika 5.
- **Ndege wadogo wa water-hole (kanga):** kwa 21.8x wana pixel 20–30 tu. Hakuna AI (site model, MegaDetector, SpeciesNet) iliyoweza kuwaona kwa ukubwa huo, hata kwa kukuza picha. Sasa nusu ya pili ya ziara ya water-hole iko kwenye **41x** (ziara ni sekunde 60). Water-hole nzima bado inaingia, na ndege wanakuwa mara 2 zaidi kwa ukubwa. Site model inaendeshwa kwa 1920 px kamili ikiwa imezoom.

### 9. Nyani, preview na camera kuendeshwa kwa mkono (19/10)
- **Kituo kipya `baboon_spot`** (preset 9: pan 0.345, camera ikiwa imeinama chini kuelekea ukuta wa mawe karibu na lodge). Patrol inakaa hapo sekunde 25 kila inapopita (`PATROL_EXTRA_STATIONS`).
- **Auto-investigate:** camera ikiwa imetulia na AI ikiona kitu cha kutiliwa shaka (≥0.45, au ≥0.35 mara mbili), mfumo unakiweka katikati na kuzoom mara 2.5 wenyewe. Imethibitishwa: nyani walioonekana kwa 3.85x walizoomwa hadi 9.6x, wakathibitishwa (0.82) na kufuatiliwa.
- **Preview:** click mnyama ili camera imweke katikati na kuzoom. **M** inahifadhi picha ya mnyama aliyekosekana (`training_data/missed/`), kwa ajili ya training ya pili. Box za kijivu "checking…" zinaonekana kwenye preview tu, si YouTube.
- **Camera ikiendeshwa kwa mkono:** patrol inasimama na kuangalia pale camera ilipoelekezwa, kisha inaendelea baada ya dakika 5 bila mkono. Kinga ya usalama haizimi mwendo milele tena (inajaribu tena baada ya dakika 1, hadi mara 3).
- **Wanyama waliotulia** hawahesabiwi tena kama "visiki" (static clutter) kama SpeciesNet au site model inawaona kama wanyama.

### 10. Patrol ya mistari miwili (19/10)
- **Kwenda kulia** (water-hole → scan_7): mstari wa juu (mbuga), kama zamani.
- **Kurudi kushoto** (scan_7 → water-hole): **mstari wa chini** upande wa nyani tu: `low_7`, `baboon_spot` (sekunde 25), `low_5` (sekunde 12). `low_2`, `low_3` na `low_4` zimeondolewa (19/10): ziko karibu na bwawa na magodoro ya wageni yalikuwa yanaonekana pembeni ya picha.
- **Faragha:** `PRIVACY_ZONES` (pan 0.425–0.745). Ndani ya eneo hilo camera hairuhusiwi kuangalia chini zaidi ya mstari wa juu. Patrol inainua camera kabla ya kuingia; mfumo ukiikuta camera huko (kwa mkono, au baada ya kuwashwa upya) unaiinua mara moja. Preview na clips zinafichwa kwa ukungu, na AI haichambui wala kuhifadhi picha hizo. Stream ya YouTube haipiti kwenye mfumo huu, kwa hiyo Privacy Mask ya camera yenyewe ndiyo kinga kamili kwake.
- Tilt ya kila kituo cha chini imechaguliwa ili **bwawa la kuogelea na viti vya wageni visionekane** (pan 0.70 imeachwa nje kwa sababu hiyo). Ukiongeza kituo, kagua picha yake kwanza.
- Wakati wa kuwasha nje ya kituo, mfumo unaangalia hapo sekunde 45 tu kabla ya patrol. Camera ikisogezwa kwa mkono wakati mfumo unaendesha, unasubiri dakika 5.
- **Tracking:** frame zisizo na MegaDetector hazihesabiwi kama "mnyama ameondoka" (site model haiwaoni nyani). Camera inaweka katikati kundi zima la wanyama.

### 11. Tracking iliyopimwa upya kabisa (19/10 jioni)
- **Mwelekeo wa pan wa tracker ulikuwa umegeuzwa** tangu mwanzo: camera ilimsukuma mnyama mbali na katikati. Sasa imepimwa kwenye camera halisi, pande zote nne, kwa 3.85x na 20x (`TRACK_PAN_IMAGE_SIGN = -1`).
- **Pulse zinapangwa kwa pixel** (`plan_pulse` kwenye `tracking/ptz_tracker.py`): kasi mbili za camera, latency na upunguzaji wa kasi kwa zoom zimepimwa. Pulse moja inasogeza 60–70% ya umbali, bila kupita lengo.
- **Zoom:** lengo ni mnyama wa ~200 px; kikomo ni ~27x (sharp). Hakuna "momentum" ya zoom tena. Zoom out inatokea tu mnyama akiwa ukingoni, anakimbia, au ni mkubwa mno. "Approach" inafanyika mara moja tu kwa kila tukio.
- **Kundi:** wanyama walio karibu wanawekwa pamoja kwenye fremu. Aliyefuatwa akijificha, tracker inamgeukia mwingine anayeonekana mara moja.
- **Baada ya kupoteza mnyama** camera inabaki sekunde 60 (`LOST_HOLD_SECONDS`) kabla ya patrol.
- **Kufuatilia mnyama anayetembea (feedforward):** tracker inapima nafasi ya mnyama duniani (pan) kila matokeo ya AI, inakadiria kasi yake, na inageuka naye kila sekunde hata kati ya matokeo. Masahihisho yanalenga mahali mnyama alipo **sasa** (si kwenye picha ya sekunde 3 nyuma). Akikosekana, camera inageukia mahali anapotarajiwa na, ikibidi, inazoom out hatua moja.
- **Idadi ya wanyama:** overlay inaonyesha "N ANIMALS IN VIEW"; `system_health.json` ina `animals_in_view`; log inaandika "Following a group of N animals".
- **Majaribio ya mfumo mzima:** `tests/test_world_scenarios.py` inaendesha control loop halisi dhidi ya ulimwengu wa kuigiza (picha, jiometri, kasi, latency na upunguzaji wa kasi kwa zoom vilivyopimwa kwenye camera halisi; AI ya sekunde 3). Hali 5: mnyama aliyesimama, anayetembea kulia/kushoto, karibu na kituo, na anayeondoka. **Kikomo kinachojulikana:** mnyama anayekimbia kwa kasi sana (~0.4°/s) kwenye ~25x bado anaweza kutoka kwenye picha kwa muda.
- **Faragha:** eneo la bwawa sasa linajumuisha pia zoom pana kuliko ~3.4x karibu na bwawa. Picha 11 za zamani za eneo hilo zimefutwa, na uvunaji wa training unaruka picha za faragha.
- **Usafishaji (19/10):** zimefutwa `.venv` ya zamani (Python 3.14, 1.7 GB), faili ya download iliyokatika (`MDV6-*.tmp`), picha za zana (`water_center_*.jpg`, `ptz_presets_low.jpg`), backup ya preset iliyopitwa na wakati, na `tools/follow_target.py`.

### 12. Ukaguzi wa detection na tracking kwa ushahidi (20/10)
- **Kati ya matokeo ya AI, tracker ilikuwa "kipofu".** Kwenye i5 hii AI inatoa matokeo kila sekunde 4–5.5, na ndiyo sababu camera "iligandia" pale ilipomwona mnyama mara ya kwanza. Sasa `tracking/box_flow.py` inamfuata mnyama kwenye kila fremu (optical flow, ~8 Hz) na inasogeza camera kwa pulse fupi kati ya matokeo. AI bado ndiyo mamlaka: kila matokeo mapya yanaweka box upya.
- **Wakati wa sekunde 60 za kusubiri baada ya kumpoteza mnyama, AI ilikuwa imezimwa.** `ptz_trace.csv` ya 19–20/10 inaonyesha `ai_busy=0` muda wote, katika matukio 4 kati ya 4. Chanzo: patrol iliyoegeshwa ("bootstrap") ilihesabiwa kama camera inayotembea. Sasa AI inaendelea kutazama, na mnyama akirudi anapatikana tena.
- **Auto-investigate:** sehemu ambayo ilithibitika kuwa na mnyama inaweza kuzoomiwa tena. Kanuni ya "si sehemu ileile kwa dakika 10" sasa inahusu miti iliyodanganya AI tu.
- **State machine moja:** PATROL → CANDIDATE → ACQUIRING → TRACKING → TARGET_LOST → RECOVERY → PATROL, pamoja na OPERATOR. Kila badiliko linaandikwa kwenye log likiwa na sababu yake (`STATE A -> B: ...`). Patrol haitumi amri yoyote wakati hali nyingine inamiliki camera.
- **Trace ya pipeline:** `logs/pipeline_trace.jsonl` ina kila matokeo ya AI (latency; box zilizogunduliwa, zilizobaki na zilizokataliwa; target na sababu ya kuchaguliwa) na kila amri ya PTZ ya tracker (chanzo, kosa kutoka katikati, ok/error). Amri ya camera ikishindwa, inaandikwa kwenye log na camera inasimamishwa.
- **Threshold ya MegaDetector haijabadilishwa, kwa sababu ya kipimo.** Kwenye picha 65 halisi (`tools/eval_set_20oct.json`), kushusha `MD_ONLY_MIN_CONFIDENCE` kutoka 0.35 hadi 0.20 hakukuongeza mnyama hata mmoja kati ya 13 waliokosekana. Box dhaifu zilikataliwa na SpeciesNet ("blank"), na kila picha ilichukua sekunde 0.5 zaidi. Nyani wadogo walio kivulini hawaonekani na site model (0/13). Training ya round 2 ndiyo inayolenga hilo; kuipima tumia `tools/eval_detection.py` (tracker ikiwa imezimwa).
- **Faili mbovu za calibration** (`ptz_presets.json`, `logs/ptz_capabilities.json`) sasa zinaandikwa kama onyo kwenye log badala ya kupuuzwa kimya. Alama ya pan inaendelea kuchukuliwa kutoka kwenye EFlip ya camera yenyewe.
- **Majaribio:** `tests/test_tracking_spec.py` ina hali 15 za ukaguzi (pamoja na 3 za ziada). Kila moja imethibitishwa kwa "mutation check": kosa la makusudi kwenye code linafanya jaribio lishindwe.
- **21/10, lock ya dakika 42 (hitilafu ya visual tracker).** Nyati alishikwa saa 06:59 kisha akatoweka. Matokeo yote 221 ya AI yaliyofuata yalikuwa matupu, lakini mfumo ulikaa `TARGET_LOST` hadi 07:41 na ulituma amri 2,223 za tilt kwenye picha isiyosogea. Chanzo: kila matokeo ya AI (hata matupu) yaliweka box upya kwenye box ya zamani na kuhesabika kama "AI imemwona". Sasa box inawekwa upya tu pale AI imemwona mnyama kweli, na visual tracker inaacha kuendesha kamera sekunde 20 (`FLOW_KEEPALIVE_SECONDS`) baada ya AI kumwona mara ya mwisho. Jaribio: `test_16`.
- **21/10, patrol kusimama dakika 5 bila target (OPERATOR ya uongo).** Saa 09:00 control loop ilikwama sekunde 3 wakati patrol ikishusha tilt kwenda `low_7`. Kamera iliendelea kusogea kwa amri yetu wenyewe, lakini loop iliporudi ilidhani mtu ameishika kamera. Sasa mfumo unalinganisha tu na nafasi iliyochukuliwa baada ya mwendo wa amri zetu kuisha. Majaribio: `test_17`, `test_17b`. Kumbuka: control loop ikikwama zaidi ya sekunde 1.5, log inaandika `Slow control tick` pamoja na sehemu iliyochelewa. Kipimo cha kwanza: kamera yenyewe inachelewa kujibu GetStatus (~2 s) karibu kila dakika, sekunde :15–:24.
- **21/10, detection wakati wa patrol.** Pause 204 za njiani (19–21/10) zilikaguliwa:
  - **Mwendo peke yake:** pause 169, **mnyama 0**. Kwa hiyo motion cue sasa inafanya kazi tu kamera ikiwa imesimama (`MOTION_CUE_WHILE_SWEEPING = False`).
  - **Box za AI:** candidates 64 zilikaguliwa kwa macho. Box ambayo SpeciesNet inaipa jina maalum (nyati, swala, ...) ilikuwa mnyama halisi mara 13 kati ya 15 mchana. Box ya jumla ("unidentified", "bird" tupu) wakati wa safari ilikuwa mnyama wa kufuatilia mara 1 tu kati ya 16 (nyingine ni miti). Usiku hakuna hata moja kati ya 11 iliyokuwa mnyama wa kufuatilia.
  - Kwa hiyo njiani kamera inasimama tu kwa box yenye jina maalum, mchana tu (`worth_a_closer_look` kwenye `main.py`). Inazoom mara moja hata ikiwa inatembea, ili nyati wa 15–22 px awe mkubwa vya kutosha kuthibitishwa.
  - Kwenye data hiyo, pause zinashuka kutoka 35 hadi 8: 6 ni halisi, na 6 kati ya 7 ya wanyama halisi wanabaki.
- **21/10, nyani waliokatiza view iliyosimama (mstari wa chini).** Kamera ikiwa imesimama na mwendo ukaonekana, picha inayofuata ya AI inaangalia crop ya 640×640 kuzunguka mwendo huo kwa resolution asili. Kwenye nyani 7 waliokosekana, MegaDetector ilipata 3 kwenye crop na 0 kwenye picha nzima. Vituo `low_5` na `low_7` sasa vinaangaliwa sekunde 12 (zamani 3, sawa na picha moja tu ya AI).
- **21/10, kundi la wanyama.** Saa 13:45, wanyama 5 walikuwa ndani ya 22% × 13% tu ya picha (kila mmoja 53–89 px), na kamera ilikuwa ikizoom nje bila sababu. Kuna sababu mbili:
  1. Kanuni ya zoom ililenga mnyama mmoja wa ~200 px.
  2. Idadi ya kundi iliruka 2↔5 kati ya matokeo ya AI, kwa hiyo kituo cha kundi kilihama na kusomeka kama "wanakimbia".

  Sasa:
  - kamera inalenga katikati ya kundi zima (pamoja na visual tracker);
  - inazoom hadi kundi lijaze 40–80% ya picha;
  - kasi inapimwa kutoka kwa mnyama mmoja aliyeshikwa;
  - approach ya kundi la mbali inaweka wote katikati.
- **21/10, zoom kupita kikomo.** Amri ya zoom ya continuous iliruka kutoka 26x hadi **42x** (lenzi ina momentum). Sasa zoom ya tracking ni hatua moja ya AbsoluteMove (×1.3) isiyopita `TRACK_MAX_ZOOM`, na zoom iliyo juu ya kikomo inarudishwa chini.
- **22/10, waterbuck kwenye mwisho wa kulia wa mstari wa chini (17:38–17:50).** Wanyama waligunduliwa vizuri, lakini mfumo ulikuwa ukiwaacha kwa sababu tatu:
  1. **Mpaka wa kulia (pan 0.24):** mara 4 kanuni ilisema "releasing lock and scanning inward". Sasa kamera inabaki ukingoni ikiwatazama. Mnyama akielekea ukingo wa picha, kamera inazoom nje ili abaki kwenye picha (approach haifanyiki mpakani).
  2. **Lock kuhama kati ya wanyama wa kundi:** kila kuhama (hadi px 856) kulisomeka kama mnyama anakimbia. Feedforward ilisukuma kamera kutoka pan 0.27 hadi 0.53 bila mnyama.
     - Sasa historia ya kasi inafutwa tu pale box mpya iko karibu zaidi na **mnyama mwingine** aliyeonekana hivi karibuni (sekunde 45) kuliko pale mnyama wetu alipotarajiwa kuwa.
     - Mnyama yuleyule akipewa track id mpya, anabaki na kasi yake. Kwenye simulation, mnyama mmoja anayetembea alipewa id 9 tofauti; kufuta kasi kila id inapobadilika kulifanya wanyama wote wanaotembea wapotee.
     - Feedforward inaruhusiwa tu baada ya kumwona mnyama mara 3, inatumia kasi ya polepole tu, na haipiti nusu ya picha bila AI kumwona tena.
  3. **Tawi:** visual tracker iliwekwa kwenye nafasi iliyotabiriwa (kichakani) ikafuata tawi linalopeperushwa. Sasa inawekwa kwenye box ya AI, ikisogezwa kwa mwendo wa kamera yenyewe.

  Pia vituo vya chini vinangoja matokeo 2 ya AI (kwenye mstari wa chini AI inachukua sekunde 10–14 kwa picha). Majaribio: `test_21`–`test_21e`.
- **Training ya round 3 (22–23/10, saa 20:30):** inaendeshwa na `training/round3.py`, na log yake ni `logs/round3.log`. Inaongeza:
  - picha 67 za kamera hii zilizothibitishwa kwa macho (`training/verified_21oct.json`, `training_data/site3v`);
  - LILA iliyolengwa (nyani, tumbili, ngiri, waterbuck ..., ndege);
  - picha mpya tangu 21/10.

  **Haikumalizika.** Harvest ilichukua kila fremu ya clips (fremu 4048), kwa hiyo labels za mwalimu zilichukua saa 10 (20:55–07:05). Dataset ikawa na tiles 26 000, karibu saa 3.1 kwa epoch moja. Dirisha la console la training lilifungwa karibu saa 07:46, katikati ya epoch 1 (exit 0xC000013A = dirisha kufungwa au Ctrl+C). Hatua ya mwisho, kuwasha tracker, haikufanyika, na tracker ilikaa imezimwa hadi 08:32. Hili halikuwa kosa la tracker yenyewe.
- **Round 3 inarudiwa (23/10, saa 20:30)** kwa `training\round3.py --reuse --deadline 05:30`:
  - inatumia upya LILA na labels za usiku uliopita;
  - inachukua fremu 1 kati ya 6 za clips na 30 % ya tiles tupu za training (validation inabaki na zote);
  - kila folder inatoa sehemu yake ya validation;
  - picha zilizokaguliwa kwa mkono zinashinda labels za mwalimu za picha zilezile: picha 14 kati ya 67 zilikuwa na label kinyume (nyani wameandikwa "hakuna kitu", mti "mnyama").

  Ukubwa uliohesabiwa ni picha 9 020 za training na 2 457 za validation, karibu dakika 73 kwa epoch. Training inasimama saa 1 kabla ya deadline (yaani 04:30), na tracker inawashwa karibu 05:15. Task inaendeshwa **bila dirisha** (`tools\run_hidden.vbs`), kwa hiyo haiwezi kufungwa kwa bahati mbaya; tracker inafunguka kwenye dirisha lake la kawaida.
- **23/10 09:28, ndege upande wa kulia (scan_7) aliachwa.**
  - Picha ambayo AI iliichambua ilipigwa kamera ikiwa njiani: pan 0.305, saa 09:28:00. Matokeo yalifika sekunde 13 baadaye, kamera ikiwa imeshasimama pan 0.282.
  - Look ilitumia picha ya **sasa** kama rejea ya box, kwa hiyo ililenga px 460 pembeni ya ndege. Ilipozoom 9.6x, ndege alikuwa nje ya picha; matokeo 2 ya AI yalikuwa matupu, na patrol ikaendelea.
  - Sasa box ya AI inatafutwa kila mara kwenye picha iliyochambuliwa, kwa kulinganisha picha (phase correlation). Kwenye picha halisi ya tukio hili ilipima mwendo wa px 460 kwa usahihi.
  - Box ya mwendo (motion) inabaki na picha ya sasa. Majaribio: `test_18e`, `test_18f`.
- **26/10 13:09, nyani hakuwekwa katikati.** Kosa lilelile la 23/10, sehemu ya pili: approach (kumkaribia mnyama mdogo aliyethibitishwa) ilitumia picha ya SASA kama rejea ya box ya picha iliyochambuliwa wakati patrol bado inatembea - ilizunguka 0.03 kupita nyani (ambaye alikuwa amesimama) na kuzoom akiwa nje ya picha. Sasa approach inatumia picha iliyochambuliwa na inasimamisha patrol kwanza. Jaribio: `test_23`.
- **23/10 12:02, kamera "inacheza" badala ya kutulia kwenye swala.**
  - Visual tracker (optical flow) inatoa box kwa aina `numpy.float32`, kwa hiyo muda wa pulse nao ulikuwa float32. `timedelta()` ya kamera iliukataa kila pulse ilipokuwa ≥ 0.6 s.
  - Masahihisho makubwa yote yalishindwa: amri 640 kati ya 21 na 23/10 (53 kati ya 106 saa 12:02–12:04). Kila kushindwa kulisimamisha kamera.
  - Sasa thamani zote zinageuzwa kuwa `float` kabla ya kwenda kwa kamera (`camera/onvif_camera.py`, `PTZTracker._command`, `follow_box`). Majaribio: `test_22`, `test_float32_pulse_from_optical_flow_is_sent`.
- **23/10 13:45–14:43, tracker iliganda saa nzima bila error.** Control loop na log zilisimama sekunde ileile, mpaka ikawashwa upya kwa mkono.
  - Chanzo kinachowezekana zaidi: QuickEdit ya cmd ilikuwa ON. Kubonyeza ndani ya dirisha kunasimamisha kila kinachoandika kwenye dirisha hilo hadi kitufe kibonyezwe.
  - Sasa tracker inazima QuickEdit ya dirisha lake inapoanza (`utils/logger.py`); imethibitishwa: mode 0x01b7, QuickEdit OFF.
- **Mlinzi (watchdog)** — task `MeliaWildlifeTrackerWatchdog`, **kila dakika 15**, `tools\tracker_watchdog.py`, log `logs\watchdog.log`:
  - ikiwa tracker haifanyi kazi, anaiwasha;
  - ikiwa imeganda (hakuna mstari mpya kwenye `logs\ptz_trace.csv` kwa dakika 5), anaiwasha upya;
  - training inayoendelea anaiacha hadi deadline ya 05:30, na ukaguzi wa model unangojewa.

  Kumzima: `schtasks /Change /TN MeliaWildlifeTrackerWatchdog /DISABLE`.
- **Round 3 ilikamilika (23–24/10) lakini haikukubaliwa.**
  - mAP50 0.690 dhidi ya 0.718. Kwa wanyama ilikuwa sawa (0.701 / 0.698), kwa ndege dhaifu (0.679 / 0.738), na box chache za uongo kwenye majani (107 / 121).
  - Sababu iliyopimwa: Ultralytics "auto" ilichagua AdamW lr 0.001667 na warm-up ya epoch 3 kati ya 7. mAP50 ilikuwa juu zaidi baada ya epoch 1 (0.759), kisha ikashuka.
- **Round 4 (25/10, saa 20:30)** — `training\round4.py`, log `logs\round4.log`:
  - fine-tune ya upole: preset `cpu_finetune` (AdamW lr 0.0003, warm-up nusu epoch, weights kila epoch); epoch yenye mAP50 bora ndiyo inapimwa;
  - Snapshot Serengeti (LILA, box za binadamu): picha 3 700 mpya — swala (Thomson's, Grant's), topi, kongoni, pofu, nyumbu, punda milia ... na ndege 1 500. Upendeleo wa picha za wanyama wadogo/wa mbali ni kwa NDEGE tu (`--small-birds-only`): picha ya kundi la swala wa mbali ilikuwa na box 2 kati ya swala ~9, na wasio na box wangefundishwa kama "si mnyama"; upakuaji una kikomo cha dakika 45 (internet ilikuwa ikikatika);
  - ukaguzi wa model sasa unapima mAP50 ya kamera hii na ya LILA kando kando, na ya kamera hii isishuke;
  - training inasimama saa 1.4 kabla ya 05:30.
- **Majina ya spishi:** confirmations 77 kati ya 321 (24 %) zilikuwa na majina ya wanyama wasiokuwepo Serengeti ("cervidae family", "bear family", "wild turkey") licha ya geofence ya TZA. Hatua inayofuata ni classifier ya spishi 61 za Serengeti, iliyofunzwa kwa crops za Snapshot Serengeti.
- **Training ya round 2 (20–21/10)** iliendeshwa kikamilifu (saa 20:30 hadi 06:32), lakini model mpya **haikukubaliwa**: mAP50 0.640 dhidi ya 0.663 ya model ya sasa, na box za uongo kwenye majani mara 3 zaidi (32 dhidi ya 10). Picha 13 za nyani (kitufe M) hazikuingia kwenye training, kwa sababu "mwalimu" (MegaDetector) hakuwaona. Ili zisaidie, zinahitaji kuwekewa box kwa mkono.

## Ndege / wanyama wadogo — hatua 2

1. **SpeciesNet** (spishi 2000+ pamoja na ndege) **imewekwa** kwenye `.venv313` (Python 3.13.15). Kwenye log lazima uone `Species classifier: Google SpeciesNet`. Kipimo kwenye camera hii: vichaka ambavyo MegaDetector iliviita "animal" (score hadi 0.65), SpeciesNet iliviita **`blank` kwa 0.95–1.00**, na sasa vinakataliwa.
2. **Model yako mwenyewe** (detector ya camera hii, darasa `animal` + `bird`): tazama `training/TRAINING_GUIDE.md`. Training inafanyika bure kwenye Google Colab GPU kwa `training/colab_train.ipynb`.

---

## Logs muhimu
| Faili | Nini |
|---|---|
| `logs/detections.csv` | wanyama **waliothibitishwa** tu, na pan/tilt/zoom |
| `logs/ptz_trace.csv` | kila sekunde: phase ya patrol, pose, amri. **Nitumie hili ukipata tatizo la camera.** |
| `logs/pipeline_trace.jsonl` | kila matokeo ya AI na kila amri ya tracker (latency, box, target na sababu, kosa kutoka katikati) |
| `logs/review/` | picha nzima + draft labels (kwa training) |
| `logs/static_false_positives.json` | "clutter" iliyojifunzwa; ifute kuanza upya |
| `logs/ptz_capabilities.json` | alama ya pan na uwezo wa PTZ uliopimwa |

`python tools/log_report.py logs/detections.csv` inaonyesha muhtasari.

## Muundo
```
camera/      ONVIF PTZ (alama ya pan, filter ya pose), focus
detection/   MegaDetector + tiles + YOLO yako, validator (masks, clutter), water zone
tracking/    patrol (water-hole kwanza), motion (closed loop), tracker
training/    harvest -> prelabel -> review -> dataset -> train (Colab) -> eval
tools/       ptz_selftest, calibrate_water_roi, validate_deployment, log_report
tests/       172 tests (pamoja na fake ya camera ya SUNBA yenye tabia zilizopimwa)
```
