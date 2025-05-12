from transformers import BertTokenizer, BertForSequenceClassification
import torch


class SentimentAnalyzer:
    def __init__(self):
        # 随机数种子
        torch.manual_seed(42)

        # 预训练模型
        self.model_name = "hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name)

        # cuda加速
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def analyze(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 计算概率
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # 0=负面(negative), 1=中性(neutral), 2=正面(positive)
        neg_prob = probabilities[0][0].item()
        neutral_prob = probabilities[0][1].item()
        pos_prob = probabilities[0][2].item()

        # 计算复合分数 (-1 到 1 的范围)
        compound = pos_prob - neg_prob

        return {
            "polarity": compound,
            "scores": {
                "neg": neg_prob,
                "neutral": neutral_prob,
                "pos": pos_prob,
                "compound": compound
            }
        }


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    examples = [
        "今天天气不好，心情很差。今天天气不好，心情很差。唉，今天天气不好，心情很差。太难过了呜呜。",
        "这部电影太糟糕了，我觉得很失望。",
        "我很开心，这个产品太棒了！我很开心，这个产品太棒了！真不错啊！想再买一个！",
        "这家餐厅的服务真的很好，食物也很美味。",
        "该公司的季度业绩超出预期，股价上涨10%。",
        "由于市场竞争激烈，该企业利润率持续下滑。"
    ]

    for text in examples:
        result = analyzer.analyze(text)
        print(f"文本: {text}")
        print(f"情感极性: {result['polarity']:.4f}")
        print(f"详细分数: {result['scores']}")
        print("-" * 50)
