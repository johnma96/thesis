# 🚀 Configuración de DVC con Google Drive (OAuth personalizado)

Este proyecto utiliza **[DVC](https://dvc.org)** para el versionamiento de datos, con **Google Drive como almacenamiento remoto**.  
El acceso se gestiona mediante una **aplicación OAuth 2.0 personalizada en Google Cloud**, lo que garantiza autenticación segura y controlada.

---

## 🧩 1. Requisitos previos

- Python 3.10+  
- Virtual environment activo (`.venv`)
- DVC instalado:

```bash
pip install "dvc[gdrive]"
```

---

## ⚙️ 2. Configuración básica de DVC

Primero, asegúrate de que el remoto de Google Drive esté definido en `.dvc/config`:

```ini
[core]
    remote = gdrive

['remote "gdrive"']
    url = gdrive://<ID_DE_TU_CARPETA_EN_GOOGLE_DRIVE>
```

> 🔹 El `<ID_DE_TU_CARPETA_EN_GOOGLE_DRIVE>` es la parte que aparece después de `folders/` en la URL de tu carpeta de Drive.  
> Ejemplo:  
> `https://drive.google.com/drive/folders/13-Epgcmqi7_UjRaSj5l8J-cGHvRM3zxO`  
> → ID: `13-Epgcmqi7_UjRaSj5l8J-cGHvRM3zxO`

---

## 🔑 3. Configurar credenciales OAuth

1. En tu **Google Cloud Console**, crea un proyecto (si no existe).  
2. Habilita la **Google Drive API**.  
3. Crea credenciales de tipo **ID de cliente de OAuth** con tipo de aplicación:  
   **“Aplicación de escritorio”**.
4. Obtendrás un **Client ID** y un **Client Secret**.  

Luego, guárdalos de forma local (no versionada):

```bash
dvc remote modify --local gdrive gdrive_client_id "TU_CLIENT_ID"
dvc remote modify --local gdrive gdrive_client_secret "TU_CLIENT_SECRET"
```

Esto genera un archivo `.dvc/config.local` como:

```ini
['remote "gdrive"']
    gdrive_client_id = TU_CLIENT_ID
    gdrive_client_secret = TU_CLIENT_SECRET
```

> ⚠️ Este archivo **no debe subirse a GitHub**, ya que contiene credenciales privadas.

---

## 👥 4. Autorizar usuarios de prueba (testers)

Si obtienes el error:

```
Error 403: access_denied
dvc-remote-storage has not completed the Google verification process
```

Debes **agregar las cuentas de Google autorizadas** para usar el proyecto:

1. En [Google Cloud Console → APIs y servicios → Pantalla de consentimiento OAuth](https://console.cloud.google.com/apis/credentials/consent)
2. En la sección **“Usuarios de prueba”**, haz clic en **“Agregar usuarios”**.
3. Agrega las direcciones de correo Gmail que usarán DVC:
   ```
   tu_correo_personal@gmail.com
   tu_correo_institucional@unal.edu.co
   ```
4. Guarda los cambios.

> 🔸 Cada cuenta agregada podrá autenticarse con este cliente OAuth sin necesidad de que la app esté verificada públicamente.  
> 🔸 Puedes tener hasta **100 testers** por proyecto.

---

## 🌐 5. Autenticación en el entorno local

Ejecuta por primera vez:

```bash
dvc push
```

Esto abrirá una ventana de navegador:
- Selecciona **“Avanzado → Ir a DVC (inseguro)”**
- Autoriza el acceso
- DVC guardará un token de sesión en:

```
C:\Users\<tu_usuario>\.dvc\tmp\gdrive-user-credentials.json
```

✅ Luego podrás subir y descargar datos sin volver a autenticarte.

---

## 🔁 6. Reautenticación o cambio de cuenta

Si deseas vincular otra cuenta de Google:
```bash
del "%HOMEPATH%\.dvc\tmp\gdrive-user-credentials.json"
dvc push
```
Luego, inicia sesión con la nueva cuenta en la ventana del navegador.

---

## 📦 7. Subir datos al remoto

Ejecuta:
```bash
dvc push
```

Verás algo como:
```
Pushing
100%|█████████████████████████████████████| 1/1 [00:10<00:00, 10.00s/file]
```

---

## 🧠 Notas finales

| Archivo | Propósito |
|----------|------------|
| `.dvc/config` | Config general del proyecto (versionada en Git) |
| `.dvc/config.local` | Credenciales privadas OAuth (no versionadas) |
| `~/.dvc/tmp/gdrive-user-credentials.json` | Token OAuth persistente (autenticación activa) |
