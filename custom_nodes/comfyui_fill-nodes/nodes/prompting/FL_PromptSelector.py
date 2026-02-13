class FL_PromptSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prepend_text": ("STRING", {"multiline": True, "default": ""}),
                "prompts": ("STRING", {"multiline": True}),
                "append_text": ("STRING", {"multiline": True, "default": ""}),
                "index": ("INT", {"default": 0, "min": 0, "max": 6969, "step": 1}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "select_prompt"
    CATEGORY = "🏵️Fill Nodes/Prompting"

    def select_prompt(self, prepend_text, prompts, append_text, index):
        prepend_text = prepend_text.strip()
        prompt_lines = prompts.split("\n")
        append_text = append_text.strip()

        num_prompts = len(prompt_lines)

        # Loop through prompts using modulo
        index = index % num_prompts

        selected_prompt = prompt_lines[index].strip()

        if prepend_text:
            selected_prompt = prepend_text + " " + selected_prompt

        if append_text:
            selected_prompt = selected_prompt + " " + append_text

        return (selected_prompt,)