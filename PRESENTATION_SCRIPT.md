# 🎤 Script พูด Present — Computer Price Forecasting and Value for Money

---

## Slide 1 — หน้าปก

> "สวัสดีครับ วันนี้ผม/พวกเราจะนำเสนอ Project ในหัวข้อ **'Computer Price Forecasting and Value for Money'**
> ซึ่งเป็นการนำ Machine Learning มาใช้วิเคราะห์และพยากรณ์ราคาคอมพิวเตอร์
> โดยในโปรเจกต์นี้เราได้แบ่งการทำงานออกเป็น 2 ส่วนหลัก คือ **Regression** และ **Classification**
> เริ่มจาก Regression ก่อนเลยครับ"

---

## Slide 2 — REGRESSION (Section Divider)

> "ส่วนแรกคือ **Regression** ซึ่งเป็นการ**พยากรณ์ราคาคอมพิวเตอร์เป็นตัวเลข** เช่น ราคา 25,000 บาท
> เราใช้ Model หลายตัว ทั้งที่ implement เองจาก scratch และใช้ Library เพื่อเปรียบเทียบผลลัพธ์ครับ"

---

## Slide 3 — All Regression Data Column

> "ก่อนอื่นมาดู **Dataset** ที่ใช้กันก่อนครับ
> ข้อมูลที่เรามีประกอบด้วย Feature หลายตัว เช่น RAM, Storage, CPU, GPU, Brand, Display Size และอื่นๆ
> ซึ่งทุก column เหล่านี้จะถูกนำมาเป็น input ในการพยากรณ์ราคาทั้งหมดครับ"

---

## Slide 4 — Regression Correlation

> "เราวิเคราะห์ **ความสัมพันธ์ (Correlation)** ระหว่าง Feature ต่างๆ กับ ราคา
> จาก Heatmap จะเห็นว่า Feature ที่มีความสัมพันธ์สูงกับราคา ได้แก่ RAM, GPU tier, และ CPU tier
> ส่วน Feature ที่สัมพันธ์น้อย เราก็ยังคงเก็บไว้เพื่อให้ Model เรียนรู้เองครับ"

---

## Slide 5 — Regression Feature Importance

> "นอกจาก Correlation เรายังดู **Feature Importance** ที่ได้จากตัว Model ด้วย
> ผลออกมาสอดคล้องกัน คือ RAM และ GPU มีผลต่อราคามากที่สุด
> ข้อมูลนี้ช่วยให้เราเข้าใจว่าคนซื้อคอมจะจ่ายเพิ่มมากที่สุดสำหรับ RAM และ GPU ครับ"

---

## Slide 6 — Gradient Boosting

> "สำหรับ Regression เราใช้ Model หลายตัวครับ ทั้ง Linear Regression, Multiple Regression, Polynomial Regression
> และ **Gradient Boosting** ซึ่งเป็นตัวที่แข็งแกร่งที่สุดในกลุ่ม
> หลักการคือ สร้าง Decision Tree ทีละต้น แต่ละต้นมาแก้ Error ของต้นก่อนหน้า สะสมกันจนได้ผลที่แม่นยำครับ"

---

## Slide 7 — Loss History and Validate History

> "นี่คือกราฟ **Loss History** ของโมเดล Regression ที่ train ด้วย parameter ต่างๆ ครับ
> ด้านซ้ายบน: learning rate = 0.01, epoch = 3000 — เห็นว่า loss ลดลงสม่ำเสมอ
> ด้านขวาบน: learning rate = 0.01, epoch = 3000 — เปรียบเทียบ train vs validation
> ด้านซ้ายล่าง: Polynomial degree = 2 — loss ลดเร็วกว่า linear
> ด้านขวาล่าง: Gradient Boosting, lr=0.05, n_estimators=300, max_depth=4 — ได้ผลดีที่สุดครับ"

---

## Slide 8 — MSE Comparison

> "เราวัดผล Regression ด้วย **MSE (Mean Squared Error)** — ยิ่งน้อยยิ่งดีครับ
> จากกราฟเปรียบเทียบทุก Model จะเห็นว่า **Gradient Boosting** ให้ค่า MSE ต่ำที่สุด
> ส่วน Linear Regression แบบง่ายมี MSE สูงกว่า เพราะไม่สามารถจับ Pattern ซับซ้อนได้ครับ"

---

## Slide 9 — R² Comparison

> "อีก Metric ที่ใช้วัดคือ **R² (R-Squared)** — ยิ่งใกล้ 1.0 ยิ่งดี แปลว่า Model อธิบาย variance ของ data ได้มาก
> **Gradient Boosting ได้ค่า R² สูงสุด** แสดงว่าสามารถพยากรณ์ราคาได้ใกล้เคียงความจริงมากที่สุดในกลุ่ม Regression ครับ"

---

## Slide 10 — CLASSIFICATION (Section Divider)

> "ส่วนที่สองคือ **Classification** ครับ
> แทนที่จะพยากรณ์ราคาเป็นตัวเลข เราเปลี่ยนเป็นการ**จัดกลุ่มราคา** เช่น ราคาต่ำ / กลาง / สูง
> เราใช้โมเดลถึง **12 ประเภท** ทั้ง Scratch และ Lib เพื่อเปรียบเทียบกันครับ"

---

## Slide 11 — ACC Comparison

> "มาดูผลแรกกันเลย นั่นคือ **Accuracy** หรือความแม่นยำโดยรวมครับ
> แกน X คือชื่อ Model, แกน Y คือค่า accuracy 0–1 สีฟ้า = Scratch, สีส้ม = Lib
> Model กลุ่ม **Tree-based** อย่าง Random Forest และ XGBoost ได้ Accuracy สูงที่สุด
> Model กลุ่ม **Clustering** อย่าง K-Means และ Agglomerative ได้ต่ำกว่า เพราะเป็น Unsupervised ไม่ได้ใช้ label ตอน train ครับ"

---

## Slide 12 — Prec/Recall/F1 Comparison

> "Slide นี้เปรียบเทียบ **Precision, Recall และ F1-Score** พร้อมกัน
> - **Precision** = จากที่ทายว่าราคาสูง ถูกกี่เปอร์เซ็นต์
> - **Recall** = จากของจริงที่ราคาสูงทั้งหมด เราตรวจเจอกี่เปอร์เซ็นต์
> - **F1** = ค่าเฉลี่ยสมดุลของทั้งสอง
>
> จะเห็นว่า Tree-based models ได้ค่าสูงทั้ง 3 ตัว ส่วน Clustering ต่ำกว่าอย่างชัดเจนครับ"

---

## Slide 13 — Specificity Comparison

> "**Specificity** หรือ True Negative Rate — วัดว่าจากคอมที่ไม่ได้ราคาสูง เราแยกออกถูกกี่เปอร์เซ็นต์
> Model ที่ Specificity สูง = ไม่ค่อย False Alarm คือไม่บอกว่าอะไรแพงทั้งที่ไม่แพง
> ผลออกมาสอดคล้องกับ Accuracy ครับ Tree-based ยังนำอยู่"

---

## Slide 14 — AUC Score Comparison

> "**AUC (Area Under the ROC Curve)** — ค่า 0.5 = สุ่มเดา, ค่า 1.0 = สมบูรณ์แบบ
> AUC วัดว่า Model สามารถแยก class ได้ดีแค่ไหนโดยไม่ขึ้นกับ threshold
> ผลจากกราฟ: **Random Forest และ XGBoost ได้ AUC ใกล้ 1.0** แสดงว่าแยก class ได้ดีมากครับ"

---

## Slide 15 — ROC Curve Comparison

> "กราฟ **ROC Curve** แสดงความสัมพันธ์ระหว่าง True Positive Rate และ False Positive Rate
> เส้นที่อยู่**ชิดมุมบนซ้ายมากที่สุด** = ดีที่สุด
> เส้น diagonal (k--) คือการ random guess ซึ่งทุก Model ของเราทำได้ดีกว่าครับ
> เส้นสีต่างๆ แทนแต่ละ Model ให้เห็นภาพรวมการแข่งขันครับ"

---

## Slide 16 — Loss/Acc Curve Comparison

> "สุดท้ายสำหรับ Classification คือกราฟ **Learning Curves**
> ด้านซ้าย Loss Curve: เห็นว่า Logistic Regression, Perceptron, SLP, MLP, SVM, XGBoost ต่างก็ค่อยๆ ลด Loss ได้
> ด้านขวา Accuracy Curve: Perceptron, SLP, MLP ค่อยๆ เพิ่ม Accuracy ขึ้นตามจำนวน Epoch
> กราฟนี้ยืนยันว่า Model เรียนรู้ได้จริง ไม่ใช่แค่จำ data ครับ"

---

## Slide 17 — THANK YOU

> "นั่นคือทั้งหมดของโปรเจกต์ **Computer Price Forecasting and Value for Money** ครับ
> สรุปคือเราได้เปรียบเทียบ Regression 6 Model และ Classification 12 Model
> ทั้ง Scratch และ Library implementation
> ผลสรุปคือ **Gradient Boosting ดีที่สุดใน Regression** และ **Random Forest / XGBoost ดีที่สุดใน Classification**
> ขอบคุณมากครับ มีคำถามอะไรไหมครับ?"

---

## 💡 Tips สำหรับตอน Q&A

| คำถาม | แนวตอบ |
|--------|--------|
| ทำไมถึงเลือก Gradient Boosting? | เพราะ sequential boosting แก้ error ได้ดีกว่า linear model สำหรับ tabular data |
| Scratch vs Lib ต่างกันยังไง? | Scratch = implement algorithm เอง ทำให้เข้าใจ math เบื้องหลัง, Lib = sklearn ที่ optimize แล้ว |
| Clustering ทำไม accuracy ต่ำ? | เพราะ unsupervised ไม่เห็น label ตอน train ต้อง map cluster กับ class ทีหลัง |
| PCA ช่วยยังไง? | ลด dimension ตัด noise ออก ทำให้ model เร็วขึ้นและบางครั้ง generalize ได้ดีขึ้น |
