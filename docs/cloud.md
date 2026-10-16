# הפעלה בענן

| אפשרות | מחיר | יתרונות | חסרונות |
|---|---|---|---|
| **Google Colab** (מומלץ להתחלה) | חינם | שרת עם GPU, ולכן הניתוח מהיר. קישור שנפתח מכל מכשיר | מפעילים בכל פעם מחדש (כ-3 דקות). הקישור מתחלף בכל הפעלה. השרת נכבה כשסוגרים את הלשונית |
| **Hugging Face Spaces** | מנוי PRO, כ-9 דולר בחודש | כתובת קבועה שתמיד זמינה | בשרת הבסיסי (CPU) הניתוח איטי |
| **המחשב שלך + קישור** | חינם | כתובת שעובדת כל עוד המחשב דולק | המחשב צריך להישאר דלוק |

---

## Google Colab (חינם, עם GPU)

המחברת `colab/basketball.ipynb` מתקינה את הכלי על שרת של Google, מפעילה את הממשק ונותנת קישור וסיסמה.

### פעם אחת
1. **מפתח קריאה ל-GitHub** (המאגר פרטי): ב-GitHub ← Settings ← Developer settings ← Personal access tokens ← **Fine-grained tokens** ← Generate new token.
   - Repository access: **Only select repositories** ← `guyos2/basketball`
   - Repository permissions ← **Contents: Read-only**
2. **פתיחת המחברת**: נכנסים ל-https://colab.research.google.com ← **File ← Open notebook** ← לשונית **GitHub**. מסמנים **Include private repos**, מאשרים גישה, בוחרים את המאגר `guyos2/basketball`, את ה-branch ואת `colab/basketball.ipynb`. אפשרות אחרת: להוריד את הקובץ מ-GitHub ולהעלות אותו בלשונית **Upload**.
3. **שמירת המפתח ב-Colab**: בסרגל הצד לוחצים על 🔑 (**Secrets**) ← Add new secret ← Name: `GITHUB_TOKEN`, Value: המפתח. מפעילים **Notebook access**.
4. **File ← Save a copy in Drive**, כדי שבפעם הבאה המחברת תחכה לכם ב-Drive.

### בכל הפעלה
1. **Runtime ← Change runtime type ← T4 GPU** ← Save.
2. מריצים את תא 1 (התקנה, כ-2 דקות) ואת תא 2 (הפעלה).
3. בסוף תא 2 מופיעים קישור וסיסמה. בכניסה מקלידים כל שם משתמש ואת הסיסמה.
4. משאירים את הלשונית של Colab פתוחה. כשהיא נסגרת, או אחרי זמן ללא שימוש, השרת נכבה והקבצים נמחקים. את התוצאות מורידים מעמוד התוצאות.

### מגבלות
- **גודל סרטון**: עד כ-95MB. הקישור עובר דרך Cloudflare Tunnel בחינם, והוא לא מעביר קבצים גדולים יותר. לקטעים של התקפה או שתיים זה מספיק. אם הסרטון גדול מדי, אפשר לקצר אותו או לייצא אותו באיכות 1080p.
- **זמן**: Colab בחינם מגביל את משך השימוש ואת זמינות ה-GPU. אם אין GPU פנוי, הכלי עובד גם בלעדיו, רק לאט יותר.

---

## Hugging Face Spaces (בתשלום)

Hugging Face דורשים היום מנוי PRO כדי ליצור Space מסוג Docker. אם יש מנוי:

1. **New Space** ← SDK: **Docker** ← Blank ← Hardware: CPU basic.
2. ב-Space ← **Settings** ← **Variables and secrets** ← New secret: `COURTVISION_PASSWORD` = הסיסמה לממשק.
3. ב-Hugging Face ← Settings ← **Access Tokens** ← טוקן מסוג **Write**.
4. במאגר ב-GitHub ← Settings ← Secrets and variables ← Actions:
   - בלשונית **Secrets**: `HF_TOKEN` = הטוקן
   - בלשונית **Variables**: `HF_SPACE` = `<שם-המשתמש>/<שם-ה-Space>`
5. כל עדכון ל-`main` מתפרסם לבד (`.github/workflows/deploy-huggingface.yml`). אפשר גם להריץ ידנית: Actions ← Deploy to Hugging Face ← Run workflow.

הממשק יהיה בכתובת `https://<שם-המשתמש>-<שם-ה-Space>.hf.space`. סרטונים ותוצאות נמחקים בהפעלה מחדש, ובכל מקרה אחרי 48 שעות (`COURTVISION_KEEP_HOURS`).

---

## המחשב שלך + קישור (חינם)

מפעילים את הממשק על המחשב (ראו README) ופותחים לו קישור ציבורי עם [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/):

```bash
COURTVISION_PASSWORD=הסיסמה-שלכם python -m courtvision serve
cloudflared tunnel --url http://localhost:8000      # בחלון נוסף; מדפיס קישור https://....trycloudflare.com
```

---

## שירותים אחרים
ה-`Dockerfile` עובד בכל שירות שמריץ קונטיינרים (Google Cloud Run, Render, Railway, Fly.io). השרת מאזין לפורט שבמשתנה `PORT`, וברירת המחדל היא 7860. צריך לפחות 4GB זיכרון. ב-Windows/Mac ב-PowerShell מגדירים סיסמה כך: `$env:COURTVISION_PASSWORD="..."`.
