# Image-Captioning-Using-CNNs-and-LSTMs

# Final Project Report

## CS 583 — Deep Learning  
**December 9, 2024**  

### Final Project Report  
**Lecturer:** Zhaozhuo Xu  
**By:** Nishant Singh (CWID:20025891)  

---

## 1. Introduction

The ability to generate meaningful and accurate captions for images is a crucial task at the intersection of computer vision and natural language processing (NLP). It has wide-ranging applications, including accessibility enhancements, automatic image tagging for social media, and improving content discovery on the web. Image captioning involves understanding the visual content of an image and translating it into coherent and contextually relevant natural language descriptions.  

Recent advancements in deep learning have enabled the development of sophisticated models capable of generating high-quality captions for images. By combining the power of Convolutional Neural Networks (CNNs) for image feature extraction and Long Short-Term Memory (LSTMs) for sequential text generation, these models can bridge the gap between vision and language.  

The aim of this project is to design and implement an image caption generation system that leverages the Flickr8k dataset, a collection of images paired with descriptive captions. Using a hybrid CNN-LSTM architecture, the project seeks to produce captions that are not only accurate but also human-like in their fluency. Such a system has the potential to transform the way visual data is interpreted and utilized in various domains, from assistive technologies to multimedia content creation.  

---

## 2. Data Collection and Processing  

### 2.1 Data Source  

The dataset used in this project is Flickr8k, a well-known benchmark for image captioning tasks. It consists of:  

- **8,000 Images:** A diverse collection of everyday scenes and objects.  
- **5 Captions per Image:** Human-annotated descriptions for each image, capturing different perspectives of the same content.  

#### Dataset Characteristics  

- The dataset includes a wide range of objects, actions, and environments, such as animals, people, and outdoor scenes.  
- Captions vary in structure and vocabulary, providing richness for training a language model.  

#### Why Flickr8k?  

Flickr8k is suitable for initial explorations in image captioning due to its manageable size. Larger datasets like MS COCO or Flickr30k might provide additional diversity but require significantly more computational resources.  

### 2.2 Data Processing  

#### Image Preprocessing  

1. **Resizing:** All images were resized to 224x224 pixels to maintain uniformity and compatibility with the pre-trained DenseNet201 model.  
2. **Normalization:** Pixel values were normalized to fall within the range [0, 1] by dividing by 255. This standardization helps stabilize the training process and improves convergence.  
3. **Feature Extraction:**  
   - Used a pre-trained DenseNet201 model, removing its classification head.  
   - Applied a Global Average Pooling (GAP) layer to extract a fixed-size, 1920-dimensional feature vector for each image.  

#### Text Preprocessing  

1. **Lowercasing:** Converted all captions to lowercase for uniformity.  
2. **Cleaning:**  
   - Removed special characters, numbers, and excessive spaces.  
   - Eliminated single-character tokens that lacked semantic value.  
3. **Start and End Tokens:** Added `<start>` and `<end>` tokens to each caption to define boundaries for the LSTM decoder.  
4. **Tokenization:**  
   - Converted each word to a numerical representation using a vocabulary of unique words.  
   - Created a mapping from words to indices and vice versa.  
5. **Padding:** Ensured all captions had the same length by padding shorter sequences with zeros.  

### 2.3 Custom Data Generation  

The Flickr8k dataset, despite being relatively small, can still pose challenges for memory-constrained environments. To address this, a custom data generator was implemented.  

#### Purpose  

1. **Efficient Memory Usage:**  
   - Processes data in batches instead of loading the entire dataset into memory.  
   - Suitable for training on systems with limited RAM.  
2. **Dynamic Data Augmentation:** Allows on-the-fly data transformation, ensuring a variety of inputs during training.  

#### Implementation Details  

- **Image Pipeline:**  
  - Reads images in batches.  
  - Extracts features using the pre-trained DenseNet201 model during preprocessing.  
  - Outputs the processed feature vectors to the training pipeline.  

- **Text Pipeline:**  
  - Tokenizes captions into sequences of indices.  
  - Pads sequences to ensure uniform length.  
  - Generates input-output pairs for the LSTM decoder:  
    - **Input:** Sequence up to the current word.  
    - **Output:** The next word in the sequence.  

#### Advantages of the Custom Generator  

1. **Scalability:** Enables processing larger datasets by batching, making it extensible to datasets like MS COCO or Flickr30k.  
2. **Efficiency:** Minimizes I/O operations and reduces computational overhead by preloading only required data.  
3. **Flexibility:** Supports dynamic adjustments such as batch size changes and data augmentation techniques.  

---

## 3. Model Development  

### 3.1 Model Architecture  

The architecture comprises two major components:  

#### Feature Extraction (Vision)  

1. **Base Model:**  
   - Used DenseNet201, a pre-trained CNN on ImageNet, to extract visual features from images.  
   - The classification head of DenseNet201 was removed, leaving only the feature extractor.  
2. **Global Average Pooling (GAP):**  
   - A GAP layer was added to reduce the spatial dimensions of extracted feature maps into a fixed-length vector (1920 dimensions).  
3. **Dense Layers:**  
   - Two fully connected layers were added to transform the feature vector into a lower-dimensional representation suitable for sequential modeling.  
   - Applied ReLU activation for non-linearity and Dropout for regularization.  

#### Sequence Generation (Text)  

1. **Embedding Layer:**  
   - Maps words in the vocabulary to dense vectors of fixed size.  
   - Captures semantic relationships between words.  
2. **LSTM Layer:**  
   - A single-layer LSTM network with 512 units was used to model sequential dependencies in text.  
   - Processes both input sequences (current words) and context from image features.  
   - Generates output probabilities for the next word in the caption sequence.  
3. **Dropout:** Applied Dropout after the LSTM to prevent overfitting.  
4. **Softmax Output Layer:** Outputs the probability distribution over the vocabulary for the next word prediction.

## 4. Results Analysis  

This section evaluates the performance of the CNN-LSTM model, highlighting the key outcomes, limitations, and areas for improvement based on training, validation, and test data results.  

### 4.1 Training and Validation Loss Trends  

The training process involved monitoring the categorical cross-entropy loss on both the training and validation datasets.  

- **Training Loss:** Decreased steadily from 4.35 to 2.71 over 50 epochs.  
- **Validation Loss:** Stabilized around 3.03 after 13 epochs, indicating convergence.  

#### Insights  

1. The validation loss closely followed the training loss, suggesting effective generalization and minimal overfitting.  
2. Early stopping was triggered when validation loss plateaued, preventing unnecessary training and reducing resource usage.  

In the training process, the loss function consistently decreased, reflecting the model's ability to learn from the data. The close alignment between training and validation loss was achieved through regularization techniques like early stopping and learning rate reduction, ensuring the model did not overfit to the training data. These techniques allowed the model to effectively generalize to unseen data while maintaining efficient convergence.  

### 4.2 Caption Generation Quality  

#### Examples of Generated Captions  

1. **Image Input:** A dog playing in water.  
   - **Generated Caption:** "A dog running through water."  
2. **Image Input:** A child in a red shirt playing outdoors.  
   - **Generated Caption:** "A child in a red shirt playing outside."  

#### Observations  

1. **Strengths:**  
   - Captions were syntactically correct and semantically relevant to the images.  
   - The model effectively identified key objects and their relationships in most cases.  
2. **Limitations:**  
   - **Repetition:** Occasionally, phrases were redundantly repeated, such as "A man standing next to a car next to a car."  
   - **Inaccuracies:** Misinterpreted object colors (e.g., "red shirt" identified as "blue shirt").  
   - **Lack of Diversity:** The model frequently generated captions with similar sentence structures.  

The model demonstrated a solid ability to generate relevant captions, correctly identifying objects and activities in the images. However, the repetitiveness in some captions and misidentification of object attributes (e.g., color errors) indicated that the model might need additional data or refinement to handle more complex or diverse visual features.  

### 4.3 Error Analysis  

#### Semantic Errors  

1. **Object Misidentification:**  
   - **Example:** A black dog was described as a "brown dog."  
   - **Cause:** Insufficient variety in training data for object color features.  
2. **Contextual Inaccuracies:**  
   - **Example:** A child holding a ball was captioned as "A child throwing a ball."  
   - **Cause:** Lack of temporal understanding in static images.  

#### Redundancy and Repetition  

1. **Observation:**  
   - Some captions repeated phrases unnecessarily (e.g., "A man sitting on a bench on a bench").  
2. **Mitigation:**  
   - Use of beam search decoding or coverage penalty during inference to discourage repetitive patterns.  

### 4.4 Comparison with Human-Generated Captions  

The generated captions were compared with human annotations to assess coherence, accuracy, and fluency.  

1. **Coherence:** The model captured the general theme of images but struggled with nuanced details.  
2. **Accuracy:** About 75% of captions correctly identified key objects and their attributes.  
3. **Fluency:** Sentence structures were grammatically correct, though somewhat formulaic.  

### 4.5 Performance Metrics  

| Metric | Value | Remarks |
|--------|-------|---------|
| **Training Loss** | 2.71 | Indicates effective learning during training. |
| **Validation Loss** | 3.03 | Reflects good generalization to unseen data. |
| **BLEU-4 Score** | 0.33 | Highlights room for improvement in sentence-level structure. |
| **Accuracy of Object Detection** | ~75% | Consistently detects major objects in images. |

### 4.6 Key Takeaways  

1. **Achievements:**  
   - Demonstrated the feasibility of the CNN-LSTM architecture for image captioning.  
   - Generated coherent and semantically relevant captions for a variety of images.  
2. **Challenges:**  
   - Over-reliance on specific sentence patterns.  
   - Limited understanding of nuanced details, such as context and relationships.  
3. **Future Work:**  
   - Incorporating attention mechanisms to improve focus on key image regions.  
   - Training on larger datasets to enhance vocabulary diversity and contextual understanding.  

---

## 5. Conclusions  

In this project, we successfully implemented a CNN-LSTM architecture to generate image captions from the Flickr8k dataset. The model demonstrated the ability to identify key objects and actions in images and generate grammatically correct, coherent captions. The training process showed steady improvement, with training and validation losses indicating effective learning and generalization. Despite these successes, the model exhibited some limitations, including redundancy in generated captions, occasional misinterpretation of object attributes, and challenges with contextual relationships between objects. These issues highlight areas where the model could be enhanced by incorporating more advanced techniques, such as attention mechanisms or training on larger, more diverse datasets.  

While the model performed well in generating relevant captions, there is room for improvement in its ability to handle complex scenes and contextual details. The results suggest that further research should focus on refining the model’s understanding of nuanced visual information and exploring alternative architectures for better performance. Overall, the project demonstrated the potential of using CNN-LSTM models for image captioning and provided a solid foundation for future advancements in this field. With improvements in model design and data diversity, image captioning systems can become even more accurate and applicable to real-world scenarios, such as assistive technologies and automated content creation.  

---  

## References  

*(List of references included in the original document remains unchanged.)*
