# Implementation & Ethics Memo

**Project: Helios AI (LIDA-based Streamlit Application)**

## Introduction

This memo reflects on the design, implementation, and ethical considerations behind **Helios AI**, a Streamlit-based prototype that uses generative AI to support data exploration and visualization. The purpose of this document is not only to describe what was built, but to explain *why* specific AI features were chosen, *how* AI was actually used during development, and *what trade-offs and risks* were considered along the way.

Throughout this project, I treated AI not as a replacement for human reasoning, but as a collaborator—both in the development process and in the final product. Building Helios AI reinforced many of the course’s core themes around responsible AI use, human oversight, and the importance of transparency when deploying generative systems.

---

## How I Actually Used AI While Building

AI played a significant role in the development of Helios AI, but in a very practical and iterative way rather than as a fully autonomous builder. I primarily used **ChatGPT** alongside **Visual Studio Code** as a “vibe coding” partner. This meant using AI to accelerate certain tasks while maintaining full responsibility for design decisions, debugging, and final implementation.

The main areas where AI assisted during development included:

* generating initial code scaffolding for Streamlit components,
* helping debug Python errors and dependency issues,
* explaining unfamiliar libraries (particularly LIDA and LiteLLM),
* suggesting UI patterns or ways to structure user flows,
* and rewriting or refining explanatory text.

However, AI-generated code was almost never used verbatim. In practice, most outputs required substantial human editing—especially when integrating multiple systems such as Streamlit, OpenAI APIs, and LIDA. Debugging environment issues, API authentication failures, visualization execution errors, and session-state logic required careful human reasoning and trial-and-error that AI alone could not resolve.

Human judgment was most important when:

* deciding *which* AI features were actually useful to users,
* diagnosing silent failures where AI suggestions were technically plausible but practically incorrect,
* and ensuring that outputs made sense in a real analytical context.

This process highlighted a key lesson from the course: AI can accelerate development, but it does not replace deep understanding of the system being built.

---

## Why the AI Feature in the Product Looks the Way It Does

The central AI feature in Helios AI is **guided data exploration** powered by LIDA. Rather than attempting to build a fully autonomous “insight engine,” I intentionally scoped the product to support users at specific decision points: summarizing a dataset, proposing analytical goals, generating visualizations, and explaining results.

I chose this feature set because it directly addresses a real pain point: many users do not struggle with *access* to data, but with knowing what questions to ask and how to interpret outputs. LIDA was a good fit because it structures AI usage around explicit steps—summarize, set goals, visualize, explain—rather than producing opaque, end-to-end answers.

Several potential AI features were intentionally excluded or scaled back. For example:

* I did not include automated “business recommendations” based on charts, because this would risk over-interpretation.
* I avoided auto-running model decisions without user confirmation.
* I limited generation to user-triggered actions rather than continuous background inference.

These choices were made to keep the AI aligned with the product’s core value proposition: **supporting, not replacing, analytical thinking**. In its current state, Helios AI connects strongly to that goal, even if some features remain partially implemented or experimental.

---

## Risks, Trade-offs, and Integrity Considerations

Building with generative AI raised several ethical and practical concerns that I actively considered.

**Privacy and data use** were a major consideration. Uploaded datasets may contain sensitive information, so the prototype is designed as a local or controlled deployment rather than a public data store. No datasets are persisted beyond a session, and AI calls are limited to schema-level summaries and code-based visualization prompts rather than raw data extraction wherever possible.

**Bias and fairness** were also important. AI-generated summaries and explanations can reflect biases present in training data or default analytical assumptions. To mitigate this, I exposed raw LIDA outputs alongside rewritten, natural-language explanations. This transparency allows users to see the underlying reasoning rather than blindly trusting a polished narrative.

**Over-reliance on AI** was a key risk I explicitly designed against. The product surfaces generated code, explanations, and evaluations, making it clear that outputs are suggestions—not facts. Users are encouraged to iterate, edit, or reject AI outputs rather than accept them as authoritative.

Finally, **academic integrity** was an important consideration in my own use of AI during development. While I used ChatGPT extensively, all architectural decisions, debugging work, and final writing were my own. AI assisted with exploration and iteration, but I remained accountable for understanding and justifying every part of the system. This aligns with the course’s emphasis on honest and transparent AI use.

---

## What I Learned About Building with Generative AI

The biggest surprise in this project was how often AI was *almost* correct but still wrong in subtle ways. Many issues—such as environment configuration, API authentication, or chart rendering, required careful human inspection that AI could not reliably perform on its own.

One major lesson I would teach another founder is this: **AI works best when constrained by clear boundaries and paired with strong human judgment**. The most successful moments in this project came when AI was used for narrow, well-defined tasks rather than open-ended problem solving.

This project also reshaped how I think about AI in future work. Rather than viewing generative AI as a shortcut, I now see it as a design material. Something that must be shaped, limited, and contextualized to create trustworthy products. As I move into future capstone or professional projects, I will be more intentional about when AI should speak, when it should stay silent, and how users are informed about its role.

---

## Conclusion

Helios AI is not a perfect or complete product, but it is an honest exploration of what it means to build with generative AI responsibly. The project reflects deliberate design choices, thoughtful constraints, and meaningful engagement with the ethical questions raised throughout the course. Most importantly, it reinforced that successful AI products are not defined by how much AI they use, but by how well they integrate human judgment, transparency, and trust.

Overall, this project has influenced how I think about AI in future academic work and potential ventures. Rather than viewing AI as a feature to be added, I now see it as an interaction layer that must be carefully designed to support human understanding, transparency, and trust.
