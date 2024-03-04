import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import tensorflow as tf
from keras import layers,models
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터셋 경로 설정
daily_conversation_folder = "C:\\Users\\duawn\\Desktop\\Senier_Project\\preprocess\\nomal"
voice_phishing_folder = "C:\\Users\\duawn\\Desktop\\Senier_Project\\preprocess\\abnomal"

# 데이터셋 로드 및 전처리
# load_dataset 함수 내에서 라벨과 파일 정보를 출력하도록 수정
def load_dataset(folder_path, label):
    texts = []
    labels = []
    filenames = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                texts.append(text)
                labels.append(label)
                filenames.append(file)

    return texts, labels, filenames


# 보이스 피싱 데이터 로드
voice_phishing_texts, voice_phishing_labels, voice_phishing_filenames = load_dataset(voice_phishing_folder, 1)

# 일상 대화 데이터 로드
daily_conversation_texts, daily_conversation_labels, daily_conversation_filenames = load_dataset(daily_conversation_folder, 0)

# 전체 데이터셋 합치기
all_texts = voice_phishing_texts + daily_conversation_texts
all_labels = voice_phishing_labels + daily_conversation_labels
all_filenames = voice_phishing_filenames + daily_conversation_filenames

# 텍스트 파일에서 불용어 읽어오기
with open("stopwords.txt", "r", encoding="utf-8") as stopword_file:
    stopwords = stopword_file.read().splitlines()

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(stop_words=stopwords)
X = vectorizer.fit_transform(all_texts)

# 각 단어에 대한 인덱스를 얻기
feature_names = vectorizer.get_feature_names_out()

# X_train에 대한 TF-IDF 가중치 확인
tfidf_weights = X.toarray()

# 특정 문서에 대한 단어들과 그에 따른 TF-IDF 가중치 저장
document_idx = 0  # 확인하고 싶은 문서의 인덱스
tfidf_weights_list = []

for word_idx in range(len(feature_names)):
    word = feature_names[word_idx]
    tfidf_weight = tfidf_weights[document_idx, word_idx]
    if tfidf_weight > 0.0:
        tfidf_weights_list.append(f"Word: {word}, TF-IDF Weight: {tfidf_weight}")

# TF-IDF 가중치를 텍스트 파일로 저장
with open("tfidf_weights.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(tfidf_weights_list))


# 라벨 인코딩
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(all_labels)

# 데이터셋 분리 (파일 이름 포함)
X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
    X, y, all_filenames, test_size=0.2, random_state=42
)



# 모델 생성
model = models.Sequential([
    layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

# 모델 컴파일
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

class_weight = {0: 1, 1: 10}  # 클래스 0에 가중치 1, 클래스 1에 가중치 10 부여
model.fit(X_train.toarray(), y_train, epochs=3, batch_size=64, validation_split=0.2, class_weight=class_weight)

# 모델 평가
test_loss, test_acc = model.evaluate(X_test.toarray(), y_test, verbose=2)
print(f"\n테스트 정확도: {test_acc}")

# 예측 수행
predictions = model.predict(X_test.toarray())

# 예측된 클래스 선택
predicted_classes = np.round(predictions).flatten()

# 예측된 인스턴스의 파일 이름, 예측된 클래스, 실제 라벨 가져오기
# results = [{"Filename": filenames_test[i], "Predicted Class": int(predicted_classes[i]), "Actual Label": int(y_test[i])}
#           for i in range(len(filenames_test))]

# 예측된 인스턴스의 파일 정보 출력
# print("예측된 인스턴스의 파일 정보:")
# for result in results:
#    print(result)


# 예측된 인스턴스의 파일 이름과 라벨 출력
# predicted_filenames = [result["Filename"] for result in results if result["Predicted Class"] == result["Actual Label"]]
# predicted_labels = [result["Predicted Class"] for result in results if result["Predicted Class"] == result["Actual Label"]]

# print("\n예측된 인스턴스의 파일 이름과 라벨:")
# for filename, label in zip(predicted_filenames, predicted_labels):
#    print(f"Filename: {filename}, Predicted Label: {label}")

# 컨퓨전 메트릭스 생성
conf_matrix = confusion_matrix(y_test, predicted_classes)

# 컨퓨전 메트릭스 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Voice Phishing'], yticklabels=['Normal', 'Voice Phishing'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 분류 보고서 출력
print("Classification Report:")
print(classification_report(y_test, predicted_classes, target_names=['Normal', 'Voice Phishing']))


# TF-IDF 정보 저장
np.save("tfidf_vectorizer.npy", vectorizer)
np.save("label_encoder.npy", label_encoder)

# 모델 저장
model.save("voice_phishing_model.h5")