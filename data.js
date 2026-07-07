// Personal Portfolio Data Configuration
// Edit this file to add/change projects, publications, blog posts, CV details, and bio content.

window.PORTFOLIO_DATA = {
  profile: {
    name: "Imrankhan",
    role: "AI Research Student",
    affiliation: "M.A.M School of Engineering",
    email: "laptop14072024@gmail.com",
    github: "https://github.com/imrancoder786",
    linkedin: "https://www.linkedin.com/in/imrankhan-developer/",
    twitter: "https://x.com/Imrankhan163180",
    location: "Thanjavur, India",
    lastUpdated: "2026",
    bio: [
      "I’m <strong>Imrankhan</strong>, an undergraduate in Computer Science at M.A.M School of Engineering (2023–2027), based in Thanjavur, India. I work on deep learning and want to become an AI researcher.",
      "My interests sit around <em>large language models</em>, <em>transformers</em>, <em>multimodal reasoning</em>, <em>self-supervised pre-training</em> (JEPA-style objectives), <em>vision models</em>, and <em>reinforcement learning</em>. I like projects where careful data handling and small-scale experiments still teach something concrete — e.g. training an EDSR super-resolution network on 300 telescope images, or reimplementing SVD from scratch to see the linear algebra fall out.",
      "I’m currently <span class=\"dotted-underline\">looking for research internships</span> for 2026. If you’re working on foundation models, multimodal learning, or efficient training and think I could contribute, I’d love to hear from you."
    ]
  },
  
  selectedWork: [
    {
      year: "2025",
      text: "EDSR super-resolution on real HSC/HST telescope imagery (PSNR 23.58 → 35.65 dB)."
    },
    {
      year: "2025",
      text: "SVD engine from scratch: image compression, LSA, collaborative filtering."
    },
    {
      year: "2025",
      text: "Winner, SRM SenthanAI Hackathon — voice-controlled desktop automation agent."
    }
  ],
  
  research: {
    intro: "I’m in the early stages of research. Below are the directions I’m actively reading and building in. This page will grow as work matures into notes, preprints, and publications.",
    interests: [
      {
        title: "Large language models & transformers",
        body: "Architectures, training dynamics, and what makes small-model reasoning fail. Interested in mechanistic explanations rather than benchmark chasing."
      },
      {
        title: "Multimodal reasoning",
        body: "Aligning vision and language for step-by-step problem solving — grounded in a working pipeline I built with Gemini for image+text tutoring queries."
      },
      {
        title: "Self-supervised pre-training (JEPA-style)",
        body: "Joint-embedding predictive objectives as an alternative to reconstruction. Reading V-JEPA and I-JEPA to understand representation quality under limited data."
      },
      {
        title: "Vision models & super-resolution",
        body: "Training under small-dataset constraints. My EDSR work on telescope imagery is the practical starting point."
      },
      {
        title: "Reinforcement learning",
        body: "Early stage — currently working through policy gradients and RLHF-style fine-tuning as tools for agentic systems."
      }
    ],
    readingList: [
      "LeCun et al., “A Path Towards Autonomous Machine Intelligence” (JEPA)",
      "Assran et al., I-JEPA / V-JEPA",
      "Sutton & Barto, <em>Reinforcement Learning: An Introduction</em>",
      "Papers with public code from ICLR / NeurIPS 2025 on efficient training"
    ]
  },
  
  projects: [
    {
      title: "Super-Resolution on Real Telescope Data",
      stack: "Python · PyTorch · EDSR · Computer Vision",
      summary: "Enhanced Deep Super-Resolution (EDSR) network to upscale real astronomical images from the HSC and HST telescopes.",
      bullets: [
        "Overcame a 300-image dataset constraint with min–max normalization and targeted augmentations (noise injection, rotations).",
        "Improved PSNR from 23.58 dB → 35.65 dB and SSIM from 0.3724 → 0.8853 over the bicubic baseline.",
        "Reduced MSE to 6.83e-4 on held-out patches."
      ],
      links: [{ label: "GitHub", href: "https://github.com/imrancoder786" }]
    },
    {
      title: "Singular Value Decomposition (SVD) Engine",
      stack: "Python · NumPy · Linear Algebra",
      summary: "A custom SVD implementation from scratch — eigen decomposition, matrix deflation, and numerical stability without ML libraries.",
      bullets: [
        "Applied Truncated SVD to image compression (CV) and Latent Semantic Analysis (NLP).",
        "Implemented collaborative filtering with matrix imputation for movie-rating prediction."
      ],
      links: [{ label: "GitHub", href: "https://github.com/imrancoder786" }]
    },
    {
      title: "Desktop Automation Agent",
      stack: "Python · Whisper · LLM APIs",
      date: "May 2025 · Hackathon winner",
      summary: "A modular voice-controlled agent capable of task planning, speech-to-text, and autonomous tool execution.",
      bullets: [
        "LLM reasoning layer for meeting summarization, scheduling, and IMAP email automation.",
        "Robust error handling for tool failures and noisy audio to keep real-time interaction stable."
      ],
      links: [{ label: "GitHub", href: "https://github.com/imrancoder786" }]
    }
  ],
  
  publications: [], // Empty state ready for future publications. Example:
  /*
  publications: [
    {
      title: "Sample Publication Title",
      authors: "Imrankhan, Jane Doe, John Smith",
      venue: "IEEE International Conference on Machine Learning (ICML)",
      year: 2026,
      links: [
        { label: "arXiv", href: "https://arxiv.org/abs/example" },
        { label: "Code", href: "https://github.com/example" }
      ],
      note: "Selected for oral presentation."
    }
  ],
  */
  
  posts: [] // Empty state ready for future blog posts. Example:
  /*
  posts: [
    {
      slug: "understanding-jepa-architectures",
      title: "Understanding Joint Embedding Predictive Architectures (JEPA)",
      date: "2026-07-07",
      summary: "An intuitive dive into Yann LeCun's JEPA model, how it differs from generative/reconstructive models, and why representation space matters.",
      body: "Joint Embedding Predictive Architectures (JEPA) represent a paradigm shift in self-supervised learning.\n\nUnlike standard autoencoders or diffusion models that try to reconstruct pixels, JEPA predicts representation embeddings of the target images in a latent space.\n\nThis prevents the model from wasting capacity on fine-grained pixel details (like leaves on a tree blowing in the wind) and instead forces it to learn higher-level semantic structures."
    }
  ]
  */
};
