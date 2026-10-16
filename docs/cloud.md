# העלאה לענן (Hugging Face Spaces)

כך הממשק זמין מכל מחשב או טלפון דרך קישור, בלי להתקין כלום. ב-Hugging Face Spaces יש שרת CPU בחינם, והוא מריץ את הממשק מתוך ה-`Dockerfile` שבמאגר.

## פעם אחת: הקמה

1. **חשבון**: נרשמים ב-https://huggingface.co.
2. **Space חדש**: לוחצים על התמונה שלכם ← **New Space**.
   - Space name: למשל `basketball`
   - SDK: **Docker** ← Blank
   - Hardware: **CPU basic** (חינם)
   - Visibility: **Public**. את הגישה סוגרים בסיסמה בשלב הבא.
3. **סיסמה לממשק**: ב-Space ← **Settings** ← **Variables and secrets** ← **New secret**:
   - Name: `COURTVISION_PASSWORD`
   - Value: הסיסמה שתרצו

   בכניסה לממשק הדפדפן יבקש שם משתמש וסיסמה. שם המשתמש לא משנה, רק הסיסמה.
4. **טוקן**: ב-Hugging Face ← **Settings** ← **Access Tokens** ← **Create new token** מסוג **Write**. מעתיקים אותו.
5. **חיבור מ-GitHub**: במאגר `guyos2/basketball` ← **Settings** ← **Secrets and variables** ← **Actions**:
   - בלשונית **Secrets** ← New repository secret: `HF_TOKEN` = הטוקן משלב 4
   - בלשונית **Variables** ← New repository variable: `HF_SPACE` = `<שם-המשתמש>/basketball`
6. **פרסום**: אחרי שהקוד נמצא ב-`main` (מיזוג ה-PR), כל עדכון ל-`main` מתפרסם לבד. אפשר גם להריץ ידנית: **Actions** ← **Deploy to Hugging Face** ← **Run workflow**.

הבנייה הראשונה לוקחת כ-5–10 דקות (רואים אותה בלשונית **Logs** של ה-Space). אחר כך הממשק זמין בכתובת:

```
https://<שם-המשתמש>-basketball.hf.space
```

## בלי GitHub Actions
אפשר לדחוף את הקבצים ל-Space ישירות מהמחשב:

```bash
git clone https://huggingface.co/spaces/<שם-המשתמש>/basketball hf-space
cp -r Dockerfile .dockerignore requirements.txt courtvision hf-space/
cp deploy/huggingface/README.md hf-space/README.md
cd hf-space && git add -A && git commit -m "Deploy" && git push
```

## מה כדאי לדעת
- **מהירות**: בשרת החינמי יש 2 מעבדים, ולכן הניתוח איטי מאשר על מחשב רגיל. לקטעים ארוכים כדאי לבחור בהגדרות המתקדמות "ניתוח כל פריים שני". אפשר לשדרג ב-Settings ← Hardware לשרת עם GPU (בתשלום לפי שעה, המחיר מופיע שם). במקרה כזה צריך לבנות עם PyTorch ל-GPU: בשורה `ARG TORCH_INDEX` ב-`Dockerfile` מחליפים את `cpu` ב-`cu124`.
- **אחסון**: הדיסק של ה-Space לא קבוע. סרטונים ותוצאות נמחקים כשה-Space מופעל מחדש, ובכל מקרה אחרי 48 שעות (`COURTVISION_KEEP_HOURS`). מה שחשוב, מורידים מהקישורים בעמוד התוצאות.
- **שינה**: Space חינמי נרדם אחרי תקופה בלי שימוש, ומתעורר בכניסה הבאה. ההתעוררות לוקחת כדקה.
- **גודל סרטון**: עד 500MB כברירת מחדל (`COURTVISION_MAX_UPLOAD_MB`).

## ענן אחר
ה-`Dockerfile` עובד בכל שירות שמריץ קונטיינרים (Google Cloud Run, Render, Railway, Fly.io). השרת מאזין לפורט שבמשתנה `PORT`, וברירת המחדל היא 7860. צריך לפחות 4GB זיכרון.
