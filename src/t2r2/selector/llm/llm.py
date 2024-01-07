from t2r2.selector.base import *
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


class LLMSelector(Selector):
    def __init__(self, prompt: str, **kwargs):
        super().__init__()
        self.prompt = prompt
        self.model = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-v0.1-AWQ", device_map="cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-v0.1-AWQ", trust_remote_code=False)

    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pipe = pipeline("text-generation", model=self.model, max_length=3000, tokenizer=self.tokenizer)
        prompt_with_data = self.prompt + ":\n" + self.df_to_str(dataset)
        prompt_template = f"{prompt_with_data}"
        return self.output_to_df(pipe(prompt_template)[0]["generated_text"])

    def df_to_str(self, dataset: pd.DataFrame) -> str:
        return dataset.apply(lambda x: f'{x["text"]},{x["label"]}', axis=1).str.cat(sep="\n")

    def output_to_df(self, output: str) -> pd.DataFrame:
        paragraphs = output.strip().split("\n")
        records = []
        for p in paragraphs:
            if "," in p:
                text, label = p.rsplit(",", 1)
                try:
                    label = int(label)
                    records.append((text, label))
                except ValueError:
                    continue
        return pd.DataFrame(records, columns=["text", "label"])
