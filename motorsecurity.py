# motorsecurity.py  — Sistema Avançado de Segurança Veicular
import sys, time, math, threading, platform, queue, os, asyncio, tempfile
from collections import deque

import numpy as np
import cv2 as cv
from PyQt5 import QtCore, QtGui, QtWidgets
import mediapipe as mp

# =========================
# ======= TTS (voz) =======
# =========================
EDGE_VOICE_SLEEP = "pt-BR-AntonioNeural"
EDGE_VOICE_SIDE  = "pt-BR-FranciscaNeural"

def _init_pyttsx3():
    try:
        import pyttsx3
        if platform.system() == "Windows":
            eng = pyttsx3.init('sapi5')
        elif platform.system() == "Darwin":
            eng = pyttsx3.init('nsss')
        else:
            eng = pyttsx3.init('espeak')
        eng.setProperty('rate', 400)
        eng.setProperty('volume', 4.0)
        return eng
    except Exception:
        return None

class TTSWorker(threading.Thread):
    """TTS industrial: tenta edge-tts (neural); se falhar, cai para pyttsx3."""
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_evt = threading.Event()
        self.q = queue.Queue()
        self.loop = asyncio.new_event_loop()
        self.edge_ok = True
        self.engine = _init_pyttsx3()

    async def _say_edge_async(self, text, voice):
        import edge_tts
        from playsound import playsound
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            mp3_path = f.name
        try:
            await edge_tts.Communicate(text, voice).save(mp3_path)
            playsound(mp3_path)
        finally:
            try: os.remove(mp3_path)
            except: pass

    def _say_fallback(self, text):
        if not self.engine:
            self.engine = _init_pyttsx3()
        if self.engine:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception:
                pass

    def speak_now(self, text, which="side"):
        self.q.put((text, which))

    def run(self):
        asyncio.set_event_loop(self.loop)
        while not self.stop_evt.is_set():
            try:
                text, which = self.q.get(timeout=0.05)
                voice = EDGE_VOICE_SIDE if which == "side" else EDGE_VOICE_SLEEP
                if self.edge_ok:
                    try:
                        self.loop.run_until_complete(self._say_edge_async(text, voice))
                    except Exception:
                        self.edge_ok = False
                        self._say_fallback(text)
                else:
                    self._say_fallback(text)
                self.q.task_done()
            except queue.Empty:
                pass

    def stop(self):
        self.stop_evt.set()
        try:
            if self.engine: self.engine.stop()
        except Exception:
            pass

# =========================
# ===== Configurações =====
# =========================
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
TARGET_FPS = 30
# tempos (indep. FPS) - CORREÇÃO: Mais rápidos
EYE_ON_SEC,  EYE_OFF_SEC  = 1.5, 0.5  # Mais rápido
SIDE_ON_SEC, SIDE_OFF_SEC = 1.5, 0.5  # Mais rápido
MOUTH_ON_SEC, MOUTH_OFF_SEC = 1.0, 0.5  # Mais rápido
PHONE_ON_SEC, PHONE_OFF_SEC = 1.0, 0.5  # Mais rápido
# thresholds - CORREÇÃO: Mais precisos
YAW_DEG_THR = 15.0  # Aumentado para menos falsos positivos
GAZE_THR    = 0.45  # Aumentado para menos falsos positivos
EAR_MIN_THR = 0.18
EAR_FACTOR  = 0.65
MOUTH_OPEN_THRESHOLD = 0.06  # CORREÇÃO: Muito mais fácil de detectar bocejo
HAND_EAR_DISTANCE_THRESHOLD = 0.12  # Mais preciso

APP_STYLES = """
* { font-family: 'Segoe UI', 'Inter', 'Roboto', Arial; }
QMainWindow { background: #0f1220; }
QStatusBar { background:#0f1220; color:#cbd5e1; border-top:1px solid #1f2a44; }
QLabel { color:#e5e7eb; }
QPushButton {
  color:#e5e7eb; background:#1b2340; border:1px solid #273154;
  padding:10px 16px; border-radius:10px;
}
QPushButton:hover { background:#22305a; }
QPushButton:pressed { background:#2a3a6e; }
"""

def np_to_qimage(frame_bgr):
    rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)

# =========================
# ===== Utilidades  =======
# =========================
class HysteresisTime:
    """Timer com histerese para evitar flickering - SEM DATACLASS"""
    def __init__(self, on_sec: float, off_sec: float):
        self.on_sec = on_sec
        self.off_sec = off_sec
        self.state = False
        self.bad_acc = 0.0
        self.good_acc = 0.0
        
    def update(self, bad: bool, dt: float) -> bool:
        if bad:
            self.bad_acc += dt
            self.good_acc = 0.0
            if not self.state and self.bad_acc >= self.on_sec:
                self.state = True
        else:
            self.good_acc += dt
            self.bad_acc = 0.0
            if self.state and self.good_acc >= self.off_sec:
                self.state = False
        return self.state

class EMA:
    def __init__(self, alpha=0.25, window=15):
        self.alpha = alpha
        self.window = deque(maxlen=window)
        self.ema = None
        
    def push(self, x: float) -> float:
        self.window.append(float(x))
        m = float(np.mean(self.window))
        self.ema = m if self.ema is None else (self.alpha * m + (1 - self.alpha) * self.ema)
        return self.ema

# =========================
# ===== Núcleo visão  =====
# =========================
class FaceSignals:
    def __init__(self):
        mp_face = mp.solutions.face_mesh
        self.mesh = mp_face.FaceMesh(
            min_detection_confidence=0.75, min_tracking_confidence=0.75,
            refine_landmarks=True, max_num_faces=1
        )
        # NOVO: Detector de mãos para celular
        self.hands = mp.solutions.hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2
        )
        
        self.eye_idx = {
            "left":  [362, 385, 387, 263, 373, 380],
            "right": [ 33, 160, 158, 133, 153, 144],
        }
        self.eye_corners = {"left": (362, 263), "right": (33, 133)}
        self.iris_idx = {"left":[468,469,470,471,472], "right":[473,474,475,476,477]}
        self.pose_idx = {"nose":1,"chin":152,"left_eye":33,"right_eye":263,"left_mouth":61,"right_mouth":291}
        
        # CORREÇÃO: Landmarks para boca (mais precisos)
        self.mouth_idx = {
            "upper": 13,  # Lábio superior
            "lower": 14   # Lábio inferior
        }
        self.ear_idx = {
            "left": 234,   # Orelha esquerda
            "right": 454   # Orelha direita
        }

    @staticmethod
    def _d(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])

    def compute_ear(self, lm):
        def ear_for(side):
            i = self.eye_idx[side]
            h  = self._d((lm[i[0]].x, lm[i[0]].y), (lm[i[3]].x, lm[i[3]].y))
            v1 = self._d((lm[i[1]].x, lm[i[1]].y), (lm[i[5]].x, lm[i[5]].y))
            v2 = self._d((lm[i[2]].x, lm[i[2]].y), (lm[i[4]].x, lm[i[4]].y))

            return (v1+v2)/(2*h) if h>1e-6 else 0.3
        le, re = ear_for("left"), ear_for("right")
        return (le+re)/2.0, le, re

    # CORREÇÃO: Cálculo muito mais sensível da abertura da boca
    def compute_mouth_openness(self, lm):
        try:
            # Pontos mais precisos para bocejo (lábio superior e inferior)
            upper_lip = lm[13]
            lower_lip = lm[14]
            
            # Abertura vertical (principal para bocejo)
            vertical_distance = abs(upper_lip.y - lower_lip.y)
            
            # DEBUG: Mostrar sempre no console para calibrar
            if vertical_distance > 0.03:  # Mostra qualquer abertura mínima
                print(f"🎯 BOCA: {vertical_distance:.3f} (thr: {MOUTH_OPEN_THRESHOLD}) - FACIL!")
            
            return vertical_distance
        except Exception as e:
            return 0.0

    # NOVO: Detectar mão próxima à orelha (celular)
    def detect_hand_near_ear(self, frame_bgr, face_landmarks):
        try:
            results = self.hands.process(cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks or not face_landmarks:
                return False

            # Posições das orelhas
            left_ear = face_landmarks[self.ear_idx["left"]]
            right_ear = face_landmarks[self.ear_idx["right"]]

            for hand_landmarks in results.multi_hand_landmarks:
                # Ponta do dedo indicador (ponto 8)
                index_tip = hand_landmarks.landmark[8]
                
                # Calcular distâncias para ambas as orelhas
                dist_left = self._d((index_tip.x, index_tip.y), (left_ear.x, left_ear.y))
                dist_right = self._d((index_tip.x, index_tip.y), (right_ear.x, right_ear.y))
                
                # Se mão próxima de qualquer orelha
                if dist_left < HAND_EAR_DISTANCE_THRESHOLD or dist_right < HAND_EAR_DISTANCE_THRESHOLD:
                    return True
                    
            return False
        except Exception:
            return False

    def estimate_pose(self, lm):
        try:
            le, re = lm[self.pose_idx["left_eye"]], lm[self.pose_idx["right_eye"]]
            lmou, rmou = lm[self.pose_idx["left_mouth"]], lm[self.pose_idx["right_mouth"]]
            nose, chin = lm[self.pose_idx["nose"]], lm[self.pose_idx["chin"]]
            eye_cx  = (le.x + re.x)*0.5
            face_cx = (eye_cx + (lmou.x + rmou.x)*0.5)*0.5
            yaw   = math.degrees(math.atan((face_cx-0.5)*2.0))
            pitch = math.degrees(math.asin(min(1.0, abs(chin.y-nose.y)*1.2)))
            roll  = math.degrees(math.atan2(re.y-le.y, re.x-le.x))
            return yaw, pitch, roll
        except Exception:
            return 0.0,0.0,0.0

    def estimate_gaze(self, lm) -> float:
        vals = []
        for side in ("left","right"):
            outer, inner = self.eye_corners[side]
            xs = [lm[i].x for i in self.iris_idx[side]]
            cx = float(np.mean(xs))
            x0, x1 = lm[outer].x, lm[inner].x
            denom = (x1 - x0)
            if abs(denom) < 1e-6: continue
            t = (cx - x0)/denom
            vals.append((t - 0.5)*2)
        return float(np.mean(vals)) if vals else 0.0

    def process(self, frame_bgr):
        res = self.mesh.process(cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB))
        if not res.multi_face_landmarks: return None
        return res.multi_face_landmarks[0].landmark

# =========================
# ====== Aplicativo  ======
# =========================
class MotorSecurityApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MotorSecurity PRO")
        self.resize(1320, 820)
        self.setStyleSheet(APP_STYLES)
        self._build_ui()

        backend = cv.CAP_DSHOW if platform.system()=="Windows" else 0
        self.cap = cv.VideoCapture(CAMERA_INDEX, backend)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.cap.set(cv.CAP_PROP_FPS, TARGET_FPS)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Erro", "Não foi possível abrir a câmera.")
            sys.exit(1)

        self.face = FaceSignals()
        self.f_ear   = EMA(0.25, 15)
        self.f_yaw   = EMA(0.25, 15)
        self.f_pitch = EMA(0.25, 15)
        self.f_gaze  = EMA(0.35, 10)
        self.f_mouth = EMA(0.3, 10)  # NOVO: Filtro para abertura da boca

        self.sw_sleep = HysteresisTime(EYE_ON_SEC,  EYE_OFF_SEC)
        self.sw_side  = HysteresisTime(SIDE_ON_SEC, SIDE_OFF_SEC)
        self.sw_mouth = HysteresisTime(MOUTH_ON_SEC, MOUTH_OFF_SEC)  # NOVO: Timer boca aberta
        self.sw_phone = HysteresisTime(PHONE_ON_SEC, PHONE_OFF_SEC)  # NOVO: Timer celular

        self.ear_thr = 0.20
        self.calibrating = True
        self.calib_end = time.time() + 4.0
        self.calib_vals = []

        self.prev_ts = time.time()
        self.fps_cnt = 0; self.fps_last = time.time(); self.fps_val = 0.0

        # TTS - CORREÇÃO: Mais rápido
        self.tts = TTSWorker(); self.tts.start()
        self._prev_sleep = False
        self._prev_side  = False
        self._prev_mouth = False  # NOVO: Estado anterior boca
        self._prev_phone = False  # NOVO: Estado anterior celular
        QtCore.QTimer.singleShot(500, lambda: self.tts.speak_now("Sistema de segurança ativado", "side"))
        # Detecção consecutiva - CORREÇÃO: Mais frames para menos falsos positivos
        self.consecutive_eye_closed = 0
        self.consecutive_distraction = 0
        self.consecutive_mouth = 0  # NOVO: Contador boca aberta
        self.consecutive_phone = 0  # NOVO: Contador celular

        # Timer de UI
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(int(1000/TARGET_FPS))

    # ---------- UI ----------
    def _build_ui(self):
        central = QtWidgets.QWidget(self); self.setCentralWidget(central)
        header = QtWidgets.QFrame(); header.setFixedHeight(78)
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #1d2342, stop:1 #3a2d6d);
                border-bottom: 1px solid #273154;
            }""")
        title = QtWidgets.QLabel("MotorSecurity PRO")
        title.setStyleSheet("color:#d6e4ff; font-size:22px; font-weight:700;")
        subt  = QtWidgets.QLabel("Monitoramento de sonolência e distração em tempo real")
        subt.setStyleSheet("color:#9fb4ff;")
        hb = QtWidgets.QVBoxLayout(header); hb.setContentsMargins(18,10,18,10); hb.addWidget(title); hb.addWidget(subt)

        self.videoLabel = QtWidgets.QLabel(); self.videoLabel.setMinimumSize(980, 560)
        self.videoLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.videoLabel.setStyleSheet("""
            QLabel { background:#0b0f1a; border:1px solid #223056; border-radius:14px; }""")

        side = QtWidgets.QFrame(); side.setFixedWidth(260)
        side.setStyleSheet("QFrame {background:#0b0f1a; border:1px solid #223056; border-radius:14px;}")
        
        # NOVO: Chips adicionais para os novos alertas
        self.chipSleep = QtWidgets.QLabel("Sonolência: NORMAL")
        self.chipSide  = QtWidgets.QLabel("Distração: NORMAL")
        self.chipMouth = QtWidgets.QLabel("Cansaço: NORMAL")      # NOVO: Chip boca aberta
        self.chipPhone = QtWidgets.QLabel("Celular: NORMAL")      # NOVO: Chip celular
        
        for chip in (self.chipSleep, self.chipSide, self.chipMouth, self.chipPhone):
            chip.setStyleSheet("color:#a7b7ff; background:#151b31; border:1px solid #233261; border-radius:10px; padding:8px 12px;")
            chip.setAlignment(QtCore.Qt.AlignCenter)

        self.btnVoice = QtWidgets.QPushButton("🔊 Testar voz")
        self.btnVoice.clicked.connect(lambda: self.tts.speak_now("Teste de voz do MotorSecurity", "side"))
        self.btnQuit = QtWidgets.QPushButton("Sair (Q)"); self.btnQuit.clicked.connect(self.close)

        grid = QtWidgets.QGridLayout(side); grid.setContentsMargins(14,14,14,14); grid.setVerticalSpacing(12)
        grid.addWidget(self._kv("Modo", "Industrial / Tempo real"), 0,0)
        grid.addWidget(self._kv("Delay alertas", "2s (on) / 0.5s (off)"), 1,0)  # Atualizado
        grid.addWidget(self._kv("Yaw thr", f">{YAW_DEG_THR:.0f}°"), 2,0)
        grid.addWidget(self._kv("Gaze thr", f">{GAZE_THR:.2f}"), 3,0)
        grid.addWidget(self._kv("Mouth thr", f">{MOUTH_OPEN_THRESHOLD:.3f}"), 4,0)  # NOVO: Threshold da boca
        grid.addWidget(self.chipSleep, 5,0); grid.addWidget(self.chipSide, 6,0)
        grid.addWidget(self.chipMouth, 7,0); grid.addWidget(self.chipPhone, 8,0)  # NOVO: Chips adicionados
        grid.addWidget(self.btnVoice, 9,0);  grid.addWidget(self.btnQuit,  10,0)

        body = QtWidgets.QHBoxLayout(); body.addWidget(self.videoLabel, 1); body.addSpacing(12); body.addWidget(side)
        self.status = QtWidgets.QStatusBar(); self.setStatusBar(self.status)
        outer = QtWidgets.QVBoxLayout(central); outer.setContentsMargins(12,12,12,12)
        outer.addWidget(header); outer.addSpacing(12); outer.addLayout(body)

    def _kv(self, k, v):
        w = QtWidgets.QWidget(); l = QtWidgets.QVBoxLayout(w); l.setContentsMargins(0,0,0,0)
        K = QtWidgets.QLabel(k); K.setStyleSheet("color:#8aa2ff; font-size:12px;")
        V = QtWidgets.QLabel(v); V.setStyleSheet("color:#e8edff; font-size:14px; font-weight:600;")
        l.addWidget(K); l.addWidget(V); return w

    # ------ Loop principal ------
    def on_tick(self):
        ok, frame = self.cap.read()
        if not ok: return
        frame = cv.flip(frame, 1)

        now = time.time()
        dt = now - self.prev_ts; self.prev_ts = now

        self.fps_cnt += 1
        if now - self.fps_last >= 1.0:
            self.fps_val = self.fps_cnt/(now - self.fps_last)
            self.fps_cnt = 0; self.fps_last = now

        lm = self.face.process(frame)
        sleep_on = side_on = mouth_on = phone_on = False  # NOVO: Variáveis para novos alertas
        ear = yaw = pitch = gaze = mouth_openness = 0.0   # NOVO: mouth_openness

        if lm is not None:
            ear_raw, _, _ = self.face.compute_ear(lm)
            yaw_raw, pitch_raw, _ = self.face.estimate_pose(lm)
            gaze_raw = self.face.estimate_gaze(lm)
            mouth_raw = self.face.compute_mouth_openness(lm)  # NOVO: Calcular abertura da boca

            ear   = self.f_ear.push(ear_raw)
            yaw   = self.f_yaw.push(yaw_raw)
            pitch = self.f_pitch.push(pitch_raw)
            gaze  = self.f_gaze.push(gaze_raw)
            mouth_openness = self.f_mouth.push(mouth_raw)  # NOVO: Filtrar abertura da boca

            if self.calibrating:
                self.calib_vals.append(ear)
                if now >= self.calib_end:
                    base = float(np.median(self.calib_vals)) if self.calib_vals else 0.28
                    self.ear_thr = float(np.clip(max(EAR_MIN_THR, base*EAR_FACTOR), 0.12, 0.35))
                    self.calibrating = False
                    self.status.showMessage(f"Calibração concluída — EAR_thr={self.ear_thr:.3f}", 4000)

            # === DETECÇÃO CONSECUTIVA MELHORADA ===
            eye_bad_original = (ear < self.ear_thr)
            side_bad_original = (abs(yaw) > YAW_DEG_THR) or (abs(gaze) > GAZE_THR)
            mouth_bad_original = (mouth_openness > MOUTH_OPEN_THRESHOLD)  # NOVO: Boca aberta
            phone_bad_original = self.face.detect_hand_near_ear(frame, lm)  # NOVO: Celular na orelha

            # CORREÇÃO: Mais frames consecutivos para menos falsos positivos
            if eye_bad_original:
                self.consecutive_eye_closed += 1
            else:
                self.consecutive_eye_closed = 0

            if side_bad_original:
                self.consecutive_distraction += 1  
            else:
                self.consecutive_distraction = 0

            # NOVO: Detecção consecutiva para boca e celular
            if mouth_bad_original:
                self.consecutive_mouth += 1
            else:
                self.consecutive_mouth = 0

            if phone_bad_original:
                self.consecutive_phone += 1
            else:
                self.consecutive_phone = 0

            # CORREÇÃO: 3 frames consecutivos para menos falsos positivos
            eye_bad = eye_bad_original and (self.consecutive_eye_closed >= 3)
            side_bad = side_bad_original and (self.consecutive_distraction >= 3)
            mouth_bad = mouth_bad_original and (self.consecutive_mouth >= 3)  # NOVO
            phone_bad = phone_bad_original and (self.consecutive_phone >= 3)  # NOVO
            # === FIM DA MELHORIA ===

            sleep_on = self.sw_sleep.update(eye_bad,  dt)
            side_on  = self.sw_side.update(side_bad,  dt)
            mouth_on = self.sw_mouth.update(mouth_bad, dt)  # NOVO: Timer boca
            phone_on = self.sw_phone.update(phone_bad, dt)  # NOVO: Timer celular

            if eye_bad:
                self._draw_eye_highlight(frame, lm)

            # CORREÇÃO: Desenhar MUITOS pontos verdes (mais de 20)
            self._draw_green_points(frame, lm)

        self._draw_debug_and_status(frame, ear, yaw, pitch, gaze, mouth_openness, sleep_on, side_on, mouth_on, phone_on)
        
        # CORREÇÃO: Banner sem caracteres especiais
        if sleep_on:
            self._banner(frame, "ALERTA! MOTORISTA DESACORDADO!", (0,0,255))
        elif side_on:
            self._banner(frame, "ATENCAO! MOTORISTA DISTRAIDO!", (0,165,255))
        elif mouth_on:  # NOVO: Banner boca aberta
            self._banner(frame, "SONO! MOTORISTA CANSADO!", (255, 165, 0))
        elif phone_on:  # NOVO: Banner celular
            self._banner(frame, "CELULAR! CELULAR DETECTADO!", (255, 0, 255))

        # Atualizar chips de status
        self.chipSleep.setText("Sonolência: " + ("ALERTA" if sleep_on else "NORMAL"))
        self.chipSleep.setStyleSheet("color:#fff; background:#a3122b; border:none; padding:8px 12px; border-radius:10px;" if sleep_on
                                     else "color:#a7b7ff; background:#151b31; border:1px solid #233261; border-radius:10px; padding:8px 12px;")
        self.chipSide.setText("Distração: " + ("ALERTA" if side_on else "NORMAL"))
        self.chipSide.setStyleSheet("color:#1b1d2b; background:#f59e0b; border:none; padding:8px 12px; border-radius:10px;" if side_on
                                     else "color:#a7b7ff; background:#151b31; border:1px solid #233261; border-radius:10px; padding:8px 12px;")
        # NOVO: Chips para novos alertas
        self.chipMouth.setText("Cansaço: " + ("ALERTA" if mouth_on else "NORMAL"))
        self.chipMouth.setStyleSheet("color:#1b1d2b; background:#ffa500; border:none; padding:8px 12px; border-radius:10px;" if mouth_on
                                     else "color:#a7b7ff; background:#151b31; border:1px solid #233261; border-radius:10px; padding:8px 12px;")
        self.chipPhone.setText("Celular: " + ("ALERTA" if phone_on else "NORMAL"))
        self.chipPhone.setStyleSheet("color:#fff; background:#8b008b; border:none; padding:8px 12px; border-radius:10px;" if phone_on
                                     else "color:#a7b7ff; background:#151b31; border:1px solid #233261; border-radius:10px; padding:8px 12px;")

        # >>> FALA SOMENTE NA ATIVAÇÃO (transição False -> True)
        if sleep_on and not self._prev_sleep:
            self.tts.speak_now("Motorista desacordado", which="sleep")
        if side_on and not self._prev_side:
            self.tts.speak_now("Motorista distraído", which="side")
        if mouth_on and not self._prev_mouth:  # NOVO: Alerta boca aberta
            self.tts.speak_now("Motorista sonolento, descanse um pouco", which="sleep")
        if phone_on and not self._prev_phone:  # NOVO: Alerta celular
            self.tts.speak_now("Saia do celular", which="side")

        self._prev_sleep = sleep_on
        self._prev_side  = side_on
        self._prev_mouth = mouth_on  # NOVO
        self._prev_phone = phone_on  # NOVO
        # <<<

        self.videoLabel.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(frame)))

    # ----- Desenho visual -----
    def _draw_eye_highlight(self, frame, lm, color=(0,0,255)):
        h, w = frame.shape[:2]
        for seq in ([362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398],
                    [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]):
            pts = np.array([[int(lm[i].x*w), int(lm[i].y*h)] for i in seq], np.int32).reshape((-1,1,2))
            cv.polylines(frame, [pts], True, color, 2)
        overlay = frame.copy()
        for group in ([362,381,263,466,386,374],[33,158,133,173,159,145]):
            poly = np.array([[int(lm[i].x*w), int(lm[i].y*h)] for i in group], np.int32)
            cv.fillPoly(overlay, [poly], color)
        cv.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # CORREÇÃO: Desenhar MUITOS pontos verdes (mais de 20 pontos estratégicos)
    def _draw_green_points(self, frame, lm, point_color=(0, 255, 0), point_size=3):
        h, w = frame.shape[:2]
        
        # MAIS DE 20 PONTOS ESTRATÉGICOS - TODAS AS REGIÕES DO ROSTO
        green_points = [
            # Olhos - 8 pontos
            33, 133, 362, 263,  # Cantos dos olhos
            468, 469, 470, 471, 472, 473, 474, 475, 476, 477,  # Íris e pupilas
            
            # Sobrancelhas - 6 pontos  
            70, 63, 105, 336, 296, 334,
            
            # Nariz - 4 pontos
            1, 4, 6, 168,
            
            # Boca - 6 pontos
            13, 14, 78, 308, 61, 291,
            
            # Contorno do rosto - 8 pontos
            10, 152, 234, 454, 138, 149, 170, 300,
            
            # Bochechas - 4 pontos
            116, 117, 346, 347,
            
            # Testa - 2 pontos
            107, 336,
            
            # Queixo - 2 pontos
            175, 199
        ]
        
        # Desenhar TODOS os pontos verdes
        for point_idx in green_points:
            if point_idx < len(lm):
                x = int(lm[point_idx].x * w)
                y = int(lm[point_idx].y * h)
                cv.circle(frame, (x, y), point_size, point_color, -1)

    def _banner(self, frame, text, color):
        overlay = frame.copy()
        cv.rectangle(overlay, (0,0), (frame.shape[1], 86), color, -1)
        cv.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        
        # Usar fonte mais compatível
        cv.putText(frame, text, (24, 56), cv.FONT_HERSHEY_SIMPLEX, 1.15, (255,255,255), 3, cv.LINE_AA)

    def _draw_debug_and_status(self, frame, ear, yaw, pitch, gaze, mouth_openness, sa, si, ma, ph):
        ok = not (sa or si or ma or ph)
        cv.putText(frame, "SISTEMA NORMAL" if ok else "ALERTA ATIVO!",
                   (22, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0) if ok else (0,0,255), 2, cv.LINE_AA)
        x0, y0 = frame.shape[1]-350, 24
        debug_lines = [
            f"EAR: {ear:.3f} thr:{self.ear_thr:.3f}",
            f"Yaw: {yaw:.1f}°  (>{YAW_DEG_THR})",
            f"Gaze: {gaze:.2f} (>{GAZE_THR})",
            f"Mouth: {mouth_openness:.3f} (>{MOUTH_OPEN_THRESHOLD})",  # NOVO
            f"Sleep:{sa}  Side:{si}",
            f"Mouth:{ma}  Phone:{ph}",  # NOVO
            f"ConsEye: {self.consecutive_eye_closed}",
            f"ConsDist: {self.consecutive_distraction}",
            f"FPS: {self.fps_val:.1f}"
        ]
        for i, s in enumerate(debug_lines):
            cv.putText(frame, s, (x0, y0+i*24), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1, cv.LINE_AA)

    # ----- Aux -----
    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape): self.close()
    def closeEvent(self, e: QtGui.QCloseEvent):
        self.timer.stop()
        if self.cap: self.cap.release()
        if self.tts: self.tts.stop()
        super().closeEvent(e)
# =========================
# ========= main ==========
# =========================
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("MotorSecurity PRO")
    app.setStyleSheet(APP_STYLES)
    w = MotorSecurityApp(); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":

    main()
