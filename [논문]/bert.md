# [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* 2018년에 구글이 공개한 사전 훈련된 모델   
* Bert는 Bidirectional Encoder Representations from Transformers 약자로 트랜스포머를 이용한 양방향 인코더 표현의 뜻.
* BERT는 Pretraning과 Fine-Tuning의 두 단계로 대규모의 텍스트 데이터를 사전 학습하고, 원하는 목적에 맞게 미세 학습을 진행하는 방식으로 매우 뛰어난 성능을 보여줌.

## Introduce
* 사전 학습된 언어 모델은 많은 자연어 처리 문제를 해결하는데 뛰어난 성능을 보임
* 사전 학습된 언어 모델을 활용할 수 있는 방법
  * 특징 기반 접근(Feature-Based Approach)
    * [ELMo](https://arxiv.org/abs/1802.05365)(2018)
    * 각 작업에 맞는 별도의 모델을 만들고, 사전 학습된 언어 모델의 출력을 추가적인 다른 모델의 입력으로 사용.
  * 미세 조정 접근(Fine-tuning Approach)
    * [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)(2018)
    * 사전 학습된 모델의 모든 파라미터를 학습 데이터에 맞춰 미세 조정. 
  * 위의 두 방법의 공통점
    * 문장을 왼쪽에서 오른쪽으로 or 오른쪽에서 왼쪽으로만 읽고 학습하는 단방향 모델.
    * 문장 전체를 이해하는데 제한적.

<br>

* BERT는 이러한 단방향 모델들의 한계를 극복하기 위해 양방향으로 학습하여 문장의 앞뒤 문맥을 고려하여 더 정확하게 이해가 가능.
* BERT는 pre-training단계에서 특히 Mask Language Model(MLM) 마스크된 언어 모델을 사용하여, 단방향의 한계를 극복
  * MLM은 입력에서 일부 토큰을 무작위로 [Mask]하고, 모델을 마스크된 단어를 문맥을 통해 예측하도록 함.  
  <img src="https://github.com/Sbeom12/study/raw/main/LLM/imgs/masked_ex.png" width ="500">

  [이미지 출처](https://velog.io/@nawnoes/%EB%82%98%EB%A7%8C%EC%9D%98-%EC%96%B8%EC%96%B4%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EA%B8%B0-Masked-Language-Model-%ED%95%99%EC%8A%B5)   
* 이뿐만 아니라 문장을 고려하여 __다음 문장 예측__ 작업도 학습.


## Bert
* BERT는 2단계로 이루어져 있다.  
<img src="https://github.com/Sbeom12/study/raw/main/LLM/imgs/Bert%EA%B8%B0%EB%B3%B8%EA%B5%AC%EC%A1%B0.png" width ="700">

  * Pre-training과 Fine-tuning 단계로 되어 있다.
  * Pre-training Unlabled data를 활용해 초기 파라미터를 설정하고,
  * Fine-tuning에서는 Pre-traning에서 학습된 초기 파라미터를 가져와 labeled data를 활용해 재학습을 진행.
  * Tasks
    * MNLI(Multi-Genre Natural Language Inference): 문장의 쌍이 주어졌을 때 그 관계를 예측하는 Task
      * 이번 주말에 친구들과 캠핑을 갔어요. 우리는 실내에 있었어요.
      * 위 두 문장이 입력으로 주어지면, 모순이라는 답변.
    * NER(Named Entity Recognition): 주어진 텍스트에서 고유명사(named entity)를 인식하고 분류하는 Task
      * 애플의 스티브 잡스는 아이폰을 발표했다. 행사는 미국 캘리포니아주 쿠퍼티노에서 열렸다.
      * 애플(기관명)의 스티브 잡스(인명)는 아이폰(제품명)을 발표했다. 행사는 미국 캘리포니아주 쿠퍼티노(지명)에서 열렸다.
    * SQuAD(Stanford Question Answering Dataset): 지문과 질문이 주어졌을때, 지문 내에서 질문에 대한 답을 찾아내는 task.
* 어떤 Task를 담당하든 BERT 모든 동일한 구조를 가진다.

## Architecture
* BERT는  multi-layer bidirectional Transformer encoder로 Transwer encoder를 여러 층 쌓은 것이다.
<img src="https://github.com/Sbeom12/study/raw/main/LLM/imgs/Transformer.png" height ="500"> 

* BERT base는 Transformer layer 12개, Head 12개, Hidden 크기를 768로 약 110M개의 파라미터로 설정
* BERT large는 Transformer layer 24개, Head 16개, Hidden 크기를 1024로 약 340M개의 파라미터로 설정.
* 특히 BERT base는 GPT와의 성능 비교를 위해 파라미터 수를 동일하게 맞춤. 

<br>

* BERT base 예시    
<img src = 'https://github.com/Sbeom12/study/assets/114134484/19d04557-9ce7-4f89-935d-891ecf2f8642' width='500'>

## Input/Ouput Representations
* BERT가 다양한 Tasks에 적용되기 위해, Input Representation이 애매하면 안된다.
* 하나의 문장 혹은 한 쌍의 문장을 하나의 토큰 시퀀스로 분명하게 표현
* 하나의 토큰 시퀀스로 표현하는 것은 매우 중요.
  * BERT는 기본적으로 NLP의 모든 Tasks에 적용할 수 있도록 만들어진 모델.
  * 기본적으로 문장이 하나만 들어올수도 2개 이상이 들어올 수 있다.
  * 모델에게 입력되는 방식이 통일되어 있어야 모델이 혼동없이 일관적으로 처리가 가능.
    * Task별로 모델에 입력되는 방식이 달라지면 안됨.
    * Task에 맞는 입력 형식을 별도로 학습이 되어야 함.
* BERT는 [CLS]토큰을 시퀀스의 시작에, [SEP]을 문장 사이 혹은, 시퀀스의 끝에 넣는 입력 형식을 통일한다.
  * 단일 문장 : [CLS] 문장 [SEP]
  * 여러 문장 : [CLS] 문장1 [SEP] 문장2 [SEP]

* Word Embedding은 [WordPiece 임베딩](https://arxiv.org/abs/1609.08144)사용
  * 단어를 의미있는 부분 단어 단위로 분할하여 임베딩하는 방법.
  * Unwanted -> 'un', 'want', 'ed'로 분할하여 임베딩 집합의 크기를 효과적으로 관리.
  * 미등록(학습데이터에 없던) 단어를 [UNK]로 처리하지 않고 분할하여 단어 분핳.
    * moderator를 'Moder', 'ator'로 분할.
* Input/Output Representations  
<img src="https://github.com/Sbeom12/study/raw/main/LLM/imgs/BERT_archi.png" width ="700">

* BERT에서 하나의 문장 혹은 한 쌍의 문장을 하나의 토큰 시퀀스로 분명하게 표현하기 위해 3가지의 Embedding vector를 Input으로 사용.
* Token Embeddings
  * 각 토큰(word or subword)을 고정 크기의 Dense벡터로 표현.
  * 토큰의 의미와 문맥적 정보를 포착.
  * 여기에서 WordPiece 토큰화가 사용된다.
* Segment Embeddings
  * BERT는 두개의 문장을 하나의 시퀀스로 처리할 수 있는데, 이때 각 단어들이 어떤 문장에 속해있는지를 나타내기 위해 사용.
  * 첫번째 문장의 토큰에는 'A'는 임베딩을, 두번째 문장 토큰에는 'B'의 임베딩을 해줌.
  * 모델이 각 토큰들이 어느 문장에 속해잇는지 알아 문장 간의 관계를 파악하는데 도움이 된다.
  * 단일 문장 Task에서는 모든 토큰에 동일한 임베딩을 함.
* Position Embeddings
  * Transformer에서도 위치 정보를 제공해야듯이, BERT에서도 Position Embeddings을 통해 각 토큰의 위치 정보를 제공함.
  * 시퀀스 내에서 토큰 순서와 구문론적 관계 등을 반영.

## Pre-training
* 시본적으로 BERT는 GPT와 ELMo와는 다르게 right-left 방향이 아니라 양방향으로 학습을 진행.

### MLM(Masked LM)
* 양방향의 학습을 위해 MLM방법을 사용.
* [Cloze 과제](https://gwern.net/doc/psychology/writing/1953-taylor.pdf)을 참고.
* 시퀀스가 입력되어 WordPiece token으로 변환되면 그 중 15%를 무작위로 마스킹, [MASK] 사용
* 마스킹된 단어를 예측. 
* 하지만 실제로 Fine-tuning을 사용할때 [MASK]가 입력되지 않아 일종의 불일치를가 존재.
* 이를 해결하기 위해 모두 [MASK] 토큰으로 대체하지 않고, 80%만 [MASK]로 대체, 10%는 무작위 토큰으로, 10%는 변경하지 않음.
* Cross_Entropy loss를 이용해서 사용.

</br>

* 단방향 VS 양방향 학습 비교  
![alt text](https://github.com/Sbeom12/study/raw/main/LLM/imgs/Pretraining_result.png)
  * MNLI: 전제와 가설이 주어지면, 어떤 관계인지 파악하는 문제.
  * 단방향 학습보다, 양방향 학습하는 것이 더 뛰어난 성능을 보이는 것을 볼 수 있다.

* MASK 토큰화 비율 학습 비교  
![alt text](https://github.com/Sbeom12/study/raw/main/LLM/imgs/table1.png)
  * MASK가 [MASK]로 대체, SAME은 그대로 유지, RND은 무작위 토큰으로 변경.
  * NER은 특히 [MASK] 토큰에 의해 영향을 크게 받을 것으로 예상해 두가지 방식으로 모두 학습 진행하여 결과 비교.
  * 특징 기반 접근법은 BERT의 마지막 4개 레이어를 연결하여 사용.
  * 최종적으로 80%, 10%, 10%으로 했을때 가장 결과가 좋았다.

### Next Sentence Prediction(NSP)
* 기존 언어 모델은 두 문장 간의 관계에 대해 직접 학습을 하지 않아 문장 관계를 이해할 수 있는 모델을 학습.
<img src="https://github.com/Sbeom12/study/raw/main/LLM/imgs/Bert%EA%B8%B0%EB%B3%B8%EA%B5%AC%EC%A1%B0.png" width ="700">

* 위 그림에서 C는 다음 문장 예측(NSP)에 사용.
* 데이터 생성
  * 학습 예제에 문장 A와 B를 선택
  * 50%의 경우, 문장 B는 문장 A 다음에 오는 문장.(IsNext로 라벨)
  * 나머지의 경우, 문장 B는 무작위로 선택된 다른 문장(NotNext로 라벨)

</br>

* NSP 학습 과정
  * C는 [CLS] 토큰의 최종 은닉 상태로 입력된 시퀀스 전체의 정보가 담겨져 있다.
  * 이 C를 이용해서 입력으로 둘어온 문장 2개가 연결되었는지(IsNext) 아닌지(NotNExt)를 예측.

* Corpus(대규모 텍스트 집합)
  * Wikipedia : 영어 위키피디아의 텍스트 데이터 사용(25억개의 단어)
  * BooksCorpus : 다양한 책에서 추출된 텍스트 데이터(8억애의 단어)
  * 약 33억 개의 단어로 이루어진 방대한 텍스트 데이터로 사전학습.

## Fine_tuning
* BERT도 기본적으로 Transformer의 Self_Attention 매커니즘을 사용하기 때문에 단일 문장이나 여러가지 문장들을 포함하는 많은 작업을 쉽게 모델링이 가능.
* 기본적으로 BERT구조를 사용하여 모든 파라미터를 각 Task에 맞게 최적화 진행.

</br>

* Input
  * text classification or sequence tagging(텍스트 분류 or 시퀀스 태깅):
    * 단일 문장 입력.
    * 텍스트 분류에서는 텍스트의 범주를 예상
    * 시눸스 태깅에서는 텍스트를 입력으로 사용하고 각 토큰의 태그를 예측.
  * Question-Passage(질문-자문)
    * 질문과 지문의 문장 쌍을 입력.
    * 질문이 입력되면 질문에 대한 답변을 지문에서 찾음.
  * Hypothesis-Premise(가설-전제)
    * 전제와 가설 문장 쌍을 입력
    * 문장의 관계가 어떠한 관계인지 확인
  * Paraphrasing(유사도)
    * 문장 쌍을 입력
    * 문장의 의미적으로 동일한지 파악.

</br>

* Output
  * token representation in sequence tagging or question answering
    * 시퀀스의 모든 토큰에 대한 모든 벡터들의 집합 사용.
    * 각 토큰의 의미 포착.
  * [CLS] representation in classification(entailment or sentiment analysis)
    * 분류 문제는 분류를 원하는 갯수(k)에 따라 classification layer를 붙여줌.
    * 문장 전체의 의미를 요약한 CLS 벡터 한개만 사용.

</br>

## Experiments
* 논문의 저자들은 BERT의 성능을 평가하기 위해 GLUE(General Language Understanding Evalution), SQuAD(Stanford Question Answering Dataset), SWAG(Situations With Adversarial Generations)를 사용.

### GlUE
* 총 9개의 Tasks로 되어 있어 모델의 언어 이해 능력을 종합적으로 평가 할 수 있다.
  * BERT에서는 WNLI의 평가지표를 사용하지 않았음.   
* 학습시 Batch_size = 32로, Fine-tuning은 3epoch만 진행.
![alt text](https://github.com/Sbeom12/study/raw/main/LLM/imgs/result_table2.png)
* MNLI(Multi-Genre Natural Language Inference): 문장의 쌍이 주어졌을 때 그 관계를 예측.(다중분류)
* QQP(Quora Question Pairs): Quora에서 수집된 질문 쌍이 의미적으로 동일한지 판단.(이진분류)
* QNLI(Question Natural Language Inference): 질문-문장 쌍이 주어졌을때, 문장이 질문에 대한 답변을 포함하는지 판단.(이진분류)
* SST-2(Stanford Sentiment Treebank): 영화 리뷰의 감정을 판단.(이진분류)
* CoLA(Corpus of Linguistic Acceptability): 문장이 문법적으로 올바른지 판단.(이진분류)
* STS-B(Semantic Textual Similarity BenchMark): 두 문장 간의 의미적 유사도를 0~5사이의 실수로 예측.(회귀)
* MRPC(Microsoft Research Paraphrase Corpus): 두 문장이 의미적으로 동일한지 판단.(이진분류)
* RTE(Recognizing Textual Entailment): 두 문장 간의 함의(포함됬는지)관계를 판단.(이진분류)

<br>

* 결과적으로 BERT가 가장 뛰어난 성능을 보여주는 것을 알 수 있다.

* 모델 크기에 대한 고찰  
![alt text](https://github.com/Sbeom12/study/raw/main/LLM/imgs/model_size_result.png)  
  * L은 layer의 수, H는 Hidden size, A는 attention heads
  * LM(ppl)은 Language Model perplexity로 언어 모델의 혼란도를 평가하는 데 사용되는 지료로 낮을 수록 좋음.
  * Fine-tuning시 모델의 크기가 클수록 효과가 좋다.

### SQuAD
* 약 100,000개의 질문/답변 쌍으로 구성된 데이터셋.
* 입력으로 질문과 정답을 포함하는 위키피디아의 한 단락이 주어졌을 때, 정답 텍스트의 범위를 예측.
* 질문에 A임베딩, 단락에는 B임베딩 사용. [CLS]질문[SEP] 단락[SEP]  

![alt text](https://github.com/Sbeom12/study/raw/main/LLM/imgs/SQuAD_result.png)
  * 기존 앙상블 모델들 보다 더 뛰어나며, 사람이 작업한것보다 성능이 더 좋았다.
  * 실제로 TriviaQA를 이용해서 미세 조정한 BERT모델들보다 더 뛰어났다.
    * TriviaQA: trivia 퀴즈라는 쉡사이트에서 수집된 QA 데이터셋.

</br>

* 추가적으로 답변이 없을 가능성을 추가하여 문제를 더 현실적으로 진행.  
![alt text](https://github.com/Sbeom12/study/raw/main/LLM/imgs/SQuAD_result2.png)

### SWAG
* 연속적인 문자열인지를 판단하는 문제.
* 기존문장과 가능한 문장 4개 입력으로 주어짐.
  * 입력 예시: [CLS] 문장 A [SEP] 문장 B1 [SEP], [CLS] 문장 A [SEP] 문장 B2 [SEP], [CLS] 문장 A [SEP] 문장 B3 [SEP], [CLS] 문장 A [SEP] 문장 B4 [SEP]  
![alt text](https://github.com/Sbeom12/study/raw/main/LLM/imgs/SWAG_result.png)


## Related Works.
* Parameters
  * BERT-base : L=12,H=768,A=12 -> 약1.1억
  * BERT-large : L=24,H=1024,A=16 ->약 3.3억

* BERT-base
<img src = 'https://github.com/Sbeom12/study/assets/114134484/19d04557-9ce7-4f89-935d-891ecf2f8642' width='500'>
<img src="https://github.com/Sbeom12/study/raw/main/LLM/imgs/BERT_archi.png" width ="700">  

  * Word_Piece : 30522
  * Hidden : 768
  * L(Transforemr Encoder) : 12개
  * A(Head) : 12개
  * 내부 신경망의 차원 : 3072

  1. Embedding Layer
    * Token Embedding으로 $30522*768 = 23,440,896$
    * Segmentation Embedding으로 $2*768 = 1,536$
    * Position Embedding으로 $512*768 = 393,216$
    * 23,835,648의 Parameter.

  2. Transformer Layer
    * query, key, value matrix : $3*768*768 =1,769,472$
    * output maxtrix : $768*768 = 589,824$
    * Feed-forward 1 : $768*3072 = 2,359,296$
    * Feed-forward 2 : $3072*768 = 2,359,296$
    * 위의 layer가 총 12개 : $12*7,077,888 = 84,934,656$

  3. Output Layer(에시로 2Class 분류)
    * $768 * 768 + 768 * 2(클래스의 개수) = 591,360$

  4. 총 파라미터 
    * $23,835,648 +84,934,656 + 591,360 = 109,361,664$로(약 1.1억)

* BERT-large
  * Word_Piece : 30522
  * Hidden : 1024
  * L(Transforemr Encoder) : 24개
  * A(Head) : 16개
  * 내부 신경망의 차원 : 4096




