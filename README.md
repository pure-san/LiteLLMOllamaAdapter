# LiteLLM Ollama Adapter

## Overview

This software enables the following:

- **Custom Ollama API Emulation**: It pretends to be an Ollama endpoint locally, but actually routes requests to **OpenRouter** through [LiteLLM](https://github.com/BerriAI/litellm).
- **No Local GPU Required**: Since OpenRouter (a litellm-supported provider) handles the model inference, you don’t need a local GPU.
- **Flexible Input Types**: Supports both text-to-text and text+image-to-text workflows.

### Why This Matters

- **Seamless Integration**: If you want to develop an application (like a LangChain workflow using ChatOllama) that relies on Ollama’s native API, this adapter allows you to do so even without native GPU hardware.

---

## System Diagram

```
Ollama application
   --->  This Software (Custom Ollama API with litellm)
        --->  OpenRouter (litellm supported provider)
```


