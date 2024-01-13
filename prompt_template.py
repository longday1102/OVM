class Prompter:
    def __init__(self):
        self.template = "Dưới đây là một Instruction mô tả nhiệm vụ. Viết một Response hoàn thành yêu cầu một cách thích hợp.\n\n ### Instruction:\n{instruction}\n\n### Response: Hãy suy nghĩ từng bước.\n"
        
    def generate_prompt(
        self,
        instruction: str,
        response: str = None,
    ) -> str:
        
        prompt = self.template.format(instruction = instruction)
        if response:
            prompt = f"{prompt}{response}"
        return prompt
    
    def get_response(self, output: str) -> str:
        parts = output.split("### Response: Hãy suy nghĩ từng bước.\n")
        if len(parts) > 1:
            return parts[1].strip()
        else:
            return ""