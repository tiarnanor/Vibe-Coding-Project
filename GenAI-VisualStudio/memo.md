#### Implementation & Ethics Memo

This memo outlines the design decisions, development process, and ethical considerations behind my LIDA-based Streamlit project. Alongside the Product Requirements Document, it explains how generative AI was used during development, why the final AI functionality was scoped the way it was, and what I learned about building interactive systems with large language models.



**How I Used AI While Building the Project**

The project was developed primarily in Visual Studio Code, using Python and Streamlit as the application framework. The core AI functionality was implemented by integrating the open-source LIDA library directly from its GitHub repository. LIDA provided the structured pipeline for dataset summarization, goal generation, visualization, explanation, evaluation, and recommendation.

In addition to these tools, I used ChatGPT extensively as a development aid throughout the build process. Rather than writing the application autonomously, ChatGPT functioned as a support tool for reasoning through implementation challenges, understanding unfamiliar APIs, debugging errors, and refining design decisions. Typical tasks where AI assistance was most useful included generating initial code scaffolds, interpreting error messages, explaining how LIDA components interact, and suggesting ways to structure Streamlit session state.

Despite this assistance, human judgment played a central role in the project. AI-generated code often required modification to work correctly in my environment, particularly when combining LIDA, Streamlit, LiteLLM, and OpenAI-compatible APIs. I frequently rewrote or simplified suggestions to better align with the project requirements and to maintain clarity and control over the system’s behavior. All final architectural decisions, integrations, and written outputs reflect my own understanding and intent.



**Why the AI Feature Is Designed This Way**

The primary AI feature of the product is an interactive data exploration assistant that helps users move from raw data to insights through visualizations. I chose this feature because it aligns closely with LIDA’s strengths and addresses a common real-world problem: enabling non-technical users to explore and understand datasets without writing code.

Rather than attempting to create a fully autonomous “AI analyst,” I intentionally designed the system to keep users in control at every stage. Users upload the data, select or refine analytical goals, trigger visualizations, and choose whether to apply AI-generated recommendations or edits. This human-in-the-loop design reflects both practical constraints and ethical considerations, ensuring that AI augments user decision-making rather than replacing it.

Some potential features were deliberately scoped out, such as persistent user accounts, automated actions without confirmation, or long-term memory across sessions. While technically possible, these features would introduce additional complexity around privacy, reliability, and governance that were not necessary for demonstrating the core value of the system within the scope of this assignment.



**Risks, Trade-offs, and Ethical Considerations**

Data privacy was a key consideration throughout development. Users may upload datasets that contain sensitive or proprietary information, so the system avoids persistent storage and processes data only within the active session. API keys are handled through environment variables and Streamlit’s secrets management rather than being hardcoded into the repository, reducing the risk of accidental exposure.

Bias and interpretability were also important concerns. Because LIDA relies on large language models to generate summaries, goals, and explanations, there is a risk that outputs may reflect biased assumptions or oversimplified narratives. To mitigate this, the system presents AI outputs as suggestions rather than facts and allows users to inspect generated code and evaluation rationales. The inclusion of an evaluation and repair stage explicitly encourages users to question and refine AI-generated results.

I also considered the risk of over-reliance on AI. Automatically generated charts and explanations can appear authoritative even when they are imperfect. By requiring explicit user actions to generate, evaluate, and apply changes, the system reinforces the idea that AI is a tool for assistance rather than a final decision-maker.

From an academic integrity perspective, I used AI tools transparently and responsibly. ChatGPT was used to support learning, debugging, and ideation, but all final work reflects my own understanding and effort. I did not submit AI-generated content without review or comprehension, and I ensured that the project demonstrates genuine engagement with the technical and conceptual material.



**What I Learned About Building with Generative AI**

One of the biggest lessons from this project was that integrating AI into a real application requires far more design and oversight than initial prototypes suggest. While generative models are powerful, they are most effective when constrained by clear interfaces, explicit user control, and robust error handling.

The most challenging aspect of the project was debugging issues that spanned multiple systems, such as environment configuration, API authentication, and deployment on Streamlit Cloud. These challenges highlighted the importance of understanding the full development stack rather than relying solely on AI-generated fixes.

If I were advising another student or founder using generative AI tools, I would emphasize treating AI as a collaborator rather than an authority. Clear prompts, careful validation, and a willingness to override or reject AI suggestions are essential. Used well, AI can significantly accelerate development, but it does not replace the need for responsibility, judgment, or ethical consideration.

Overall, this project has influenced how I think about AI in future academic work and potential ventures. Rather than viewing AI as a feature to be added, I now see it as an interaction layer that must be carefully designed to support human understanding, transparency, and trust.
