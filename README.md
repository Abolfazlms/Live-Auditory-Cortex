# Real-Time Auditory Cortex Simulation using Self-Organizing Maps  
### شبیه‌سازی زندهٔ قشر شنوایی با استفاده از نقشهٔ خودسازمان‌ده (SOM)

**Author:** Abolfazl Mastaalizadeh  
**Version:** 1.2.0  
**Date:** November 2025  
**License:** MIT  
**Repository:** https://github.com/[yourusername]/auditory-som-live  

---

## معرفی پروژه
این پروژه یک شبیه‌ساز بلادرنگ و تعاملی از نحوهٔ خودسازمان‌یابی قشر شنوایی اولیه (A1) در مغز است.  
با استفاده از ورودی زندهٔ میکروفون و یک شبکهٔ **Self-Organizing Map (SOM)** ۲۰×۲۰، برنامه به‌صورت لحظه‌ای ویژگی‌های صوتی را استخراج کرده، شبکه را آموزش می‌دهد و شکل‌گیری تدریجی نقشهٔ تونوتوپیک (Tonotopic Map) را به‌صورت زنده نمایش می‌دهد — دقیقاً شبیه به آنچه در سیستم شنوایی زیستی رخ می‌دهد.

---

## ویژگی‌های کلیدی
- **ورودی کاملاً زنده** از میکروفون (بدون نیاز به فایل صوتی)
- استخراج بلادرنگ ویژگی‌های شنیداری:
  - انرژی لگاریتمی باند پایین و میانی
  - شدت کلی سیگنال
- آموزش **آنلاین و افزایشی** SOM (یک نمونه در هر مرحله)
- داشبورد زنده با ۵ نمایشگر همزمان:
  - موج صوتی (۴ ثانیه آخر)
  - اسپکتروگرام زنده
  - نمودار زمانی ویژگی‌ها
  - نقشهٔ تونوتوپیک دوبعدی در حال تکامل
  - پراکندگی سه‌بعدی وزن‌های نورون‌ها
- توقف ایمن با فشار دادن کلید **`q`** روی پنجره
- ذخیرهٔ خودکار وزن‌های نهایی و تصویر خلاصه با کیفیت بالا

---

## پشتوانه علمی
پروژه بر اساس اصول زیر توسعه یافته است:
- Kohonen, T. (1982). *Self-Organizing Maps*
- سازمان تونوتوپیک و دوره‌توپیک در قشر شنوایی پستانداران
- یادگیری بدون ناظر در سیستم‌های حسی زیستی
- مدل‌های محاسباتی شنوایی (Computational Auditory Scene Analysis)

---

## نصب و اجرا

### ۱) نصب وابستگی‌ها
```bash
pip install -r requirements.txt
۲) فایل requirements.txt
txtminisom>=2.3.0
pyaudio>=0.2.14
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.8.0
نکته: در ویندوز ممکن است نیاز به نصب PyAudio از طریق wheel داشته باشید.
اجرا
Bashpython main.py
پس از اجرا:

برنامه شروع به شنود میکروفون می‌کند
پنجرهٔ داشبورد ظاهر می‌شود
با کلیک روی پنجره و فشار دادن q، برنامه متوقف شده و نتایج ذخیره می‌شوند


خروجی‌ها
تمام نتایج در پوشهٔ auditory_som_results/ ذخیره می‌شوند:

















فایلتوضیحاتfinal_som_weights.npyوزن‌های نهایی SOM (آرایهٔ ۲۰×۲۰×۳)final_tonotopic_map.pngتصویر نهایی نقشهٔ تونوتوپیک + نمودار ویژگی‌ها

معماری سیستم
#mermaid-diagram-mermaid-eqfyni2{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#000000;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#mermaid-diagram-mermaid-eqfyni2 .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#mermaid-diagram-mermaid-eqfyni2 .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#mermaid-diagram-mermaid-eqfyni2 .error-icon{fill:#552222;}#mermaid-diagram-mermaid-eqfyni2 .error-text{fill:#552222;stroke:#552222;}#mermaid-diagram-mermaid-eqfyni2 .edge-thickness-normal{stroke-width:1px;}#mermaid-diagram-mermaid-eqfyni2 .edge-thickness-thick{stroke-width:3.5px;}#mermaid-diagram-mermaid-eqfyni2 .edge-pattern-solid{stroke-dasharray:0;}#mermaid-diagram-mermaid-eqfyni2 .edge-thickness-invisible{stroke-width:0;fill:none;}#mermaid-diagram-mermaid-eqfyni2 .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-diagram-mermaid-eqfyni2 .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-diagram-mermaid-eqfyni2 .marker{fill:#666;stroke:#666;}#mermaid-diagram-mermaid-eqfyni2 .marker.cross{stroke:#666;}#mermaid-diagram-mermaid-eqfyni2 svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#mermaid-diagram-mermaid-eqfyni2 p{margin:0;}#mermaid-diagram-mermaid-eqfyni2 .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#000000;}#mermaid-diagram-mermaid-eqfyni2 .cluster-label text{fill:#333;}#mermaid-diagram-mermaid-eqfyni2 .cluster-label span{color:#333;}#mermaid-diagram-mermaid-eqfyni2 .cluster-label span p{background-color:transparent;}#mermaid-diagram-mermaid-eqfyni2 .label text,#mermaid-diagram-mermaid-eqfyni2 span{fill:#000000;color:#000000;}#mermaid-diagram-mermaid-eqfyni2 .node rect,#mermaid-diagram-mermaid-eqfyni2 .node circle,#mermaid-diagram-mermaid-eqfyni2 .node ellipse,#mermaid-diagram-mermaid-eqfyni2 .node polygon,#mermaid-diagram-mermaid-eqfyni2 .node path{fill:#eee;stroke:#999;stroke-width:1px;}#mermaid-diagram-mermaid-eqfyni2 .rough-node .label text,#mermaid-diagram-mermaid-eqfyni2 .node .label text,#mermaid-diagram-mermaid-eqfyni2 .image-shape .label,#mermaid-diagram-mermaid-eqfyni2 .icon-shape .label{text-anchor:middle;}#mermaid-diagram-mermaid-eqfyni2 .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#mermaid-diagram-mermaid-eqfyni2 .rough-node .label,#mermaid-diagram-mermaid-eqfyni2 .node .label,#mermaid-diagram-mermaid-eqfyni2 .image-shape .label,#mermaid-diagram-mermaid-eqfyni2 .icon-shape .label{text-align:center;}#mermaid-diagram-mermaid-eqfyni2 .node.clickable{cursor:pointer;}#mermaid-diagram-mermaid-eqfyni2 .root .anchor path{fill:#666!important;stroke-width:0;stroke:#666;}#mermaid-diagram-mermaid-eqfyni2 .arrowheadPath{fill:#333333;}#mermaid-diagram-mermaid-eqfyni2 .edgePath .path{stroke:#666;stroke-width:2.0px;}#mermaid-diagram-mermaid-eqfyni2 .flowchart-link{stroke:#666;fill:none;}#mermaid-diagram-mermaid-eqfyni2 .edgeLabel{background-color:white;text-align:center;}#mermaid-diagram-mermaid-eqfyni2 .edgeLabel p{background-color:white;}#mermaid-diagram-mermaid-eqfyni2 .edgeLabel rect{opacity:0.5;background-color:white;fill:white;}#mermaid-diagram-mermaid-eqfyni2 .labelBkg{background-color:rgba(255, 255, 255, 0.5);}#mermaid-diagram-mermaid-eqfyni2 .cluster rect{fill:hsl(0, 0%, 98.9215686275%);stroke:#707070;stroke-width:1px;}#mermaid-diagram-mermaid-eqfyni2 .cluster text{fill:#333;}#mermaid-diagram-mermaid-eqfyni2 .cluster span{color:#333;}#mermaid-diagram-mermaid-eqfyni2 div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(-160, 0%, 93.3333333333%);border:1px solid #707070;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-diagram-mermaid-eqfyni2 .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#000000;}#mermaid-diagram-mermaid-eqfyni2 rect.text{fill:none;stroke-width:0;}#mermaid-diagram-mermaid-eqfyni2 .icon-shape,#mermaid-diagram-mermaid-eqfyni2 .image-shape{background-color:white;text-align:center;}#mermaid-diagram-mermaid-eqfyni2 .icon-shape p,#mermaid-diagram-mermaid-eqfyni2 .image-shape p{background-color:white;padding:2px;}#mermaid-diagram-mermaid-eqfyni2 .icon-shape rect,#mermaid-diagram-mermaid-eqfyni2 .image-shape rect{opacity:0.5;background-color:white;fill:white;}#mermaid-diagram-mermaid-eqfyni2 :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}Microphone InputAudio Chunk
2048 samplesFFT → Band Energies
Low / Mid + Intensitylog(low), log(mid), intensity20×20 Self-Organizing Map
Online TrainingLive Visualization Dashboard

ساختار پروژه
textauditory-som-live/
├── main.py                     # کد اصلی
├── requirements.txt
├── README.md
└── auditory_som_results/       # خروجی‌ها (خودکار ساخته می‌شود)

مسیرهای توسعه آینده

افزودن Mel-Spectrogram و MFCC به عنوان ویژگی
پیاده‌سازی Growing SOM یا Hierarchical SOM
نسخهٔ PyTorch با GPU acceleration
ضبط ویدیو از داشبورد زنده
تحلیل خودکار گرادیان تونوتوپیک
وب‌اپلیکیشن با WebRTC + WebAssembly


مشارکت
Pull Request و Issue کاملاً باز و خوش‌آمد است!
اگر ایده، باگ یا پیشنهادی دارید، با کمال میل منتظرتون هستم

لایسنس
این پروژه تحت لایسنس MIT منتشر شده است.
استفاده، تغییر و توزیع آزاد است.

Developed with passion for Neuroscience & AI
Abolfazl Mastaalizadeh — November 2025