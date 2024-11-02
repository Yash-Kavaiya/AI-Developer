# Generative AI Service

OCI Generative Al Service
• Pretrained FoundationalModels
• Prompt Engineering and LLM Customization
• Fine-tuning and Inference
• Dedicated Al Clusters
• Generative Al Security Architecture

# OCI Generative AI Service

The **OCI Generative AI Service** is a fully managed service offered by Oracle Cloud Infrastructure (OCI) that enables developers and businesses to build generative AI applications using customizable Large Language Models (LLMs). These models are accessible through a single API and can be fine-tuned to meet specific needs. The service leverages GPU-based clusters to handle large-scale AI workloads, ensuring high performance for both training and inference.

## Key Features

### 1. Choice of Models
OCI offers a selection of high-performing, pretrained foundational models from industry leaders, **Meta** and **Cohere**. These models support various generative AI applications, such as chatbots, information retrieval, and text generation.

### 2. Flexible Fine-tuning
OCI Generative AI Service allows users to create custom models by fine-tuning foundational models with their own datasets. Fine-tuning enables tailored model performance, improving relevance and accuracy for specific tasks and industries.

### 3. Dedicated AI Clusters
The service provides dedicated AI clusters powered by GPUs to host fine-tuning and inference tasks. These clusters ensure isolation and dedicated resources, maintaining optimal performance and security for your AI workloads.

---

## How OCI Generative AI Service Works

OCI’s Generative AI Service is built to understand, generate, and process human language at scale, supporting various use cases such as:

- **Chat**: Enables conversational AI applications by providing natural, responsive dialogue capabilities.
- **Text Generation**: Generates human-like text, making it ideal for content creation, summarization, and more.
- **Information Retrieval**: Retrieves relevant information from vast data sources, enhancing knowledge-based applications.
- **Semantic Search**: Uses embeddings to provide contextual search capabilities.

## Supported Model Types

The OCI Generative AI Service includes multiple pretrained foundational models tailored to different use cases:

### 1. **Chat Models**
   - Respond to questions with conversational answers, suitable for customer service, virtual assistants, and interactive applications.

### 2. **Instruction-following Models**
   - Designed to follow task-specific instructions, enabling automation in complex, structured workflows.

### 3. **Embedding Models**
   - Convert text into vector embeddings, enabling similarity searches and recommendations. OCI offers multilingual embedding models, making it suitable for applications that support multiple languages.

---

### Specific Models and Their Uses

| Model Name                | Type                  | Provider   | Use Case                                         |
|---------------------------|-----------------------|------------|--------------------------------------------------|
| `command-r-plus`          | Chat, Text Generation | Cohere     | Conversational applications, question answering   |
| `command-r-16k`           | Chat, Text Generation | Cohere     | Extended dialogue applications with large context |
| `lama 3-70b-instruct`     | Instruction-following | Meta       | Instruction-specific workflows                    |
| `embed-english-v3.0`      | Embedding             | Cohere     | Semantic search in English                        |
| `embed-multilingual-v3.0` | Embedding             | Cohere     | Multilingual semantic search                      |

## Fine-tuning Options

Fine-tuning enables the customization of pretrained foundational models to improve performance on specific tasks. This customization is ideal when:

- The pretrained model doesn’t perform well on the task.
- You want the model to learn new, domain-specific information.

OCI uses **T-Few fine-tuning** (from Cohere), which is a highly efficient method for customizing models. Fine-tuning enhances model accuracy, efficiency, and relevance, and allows organizations to optimize their AI for industry-specific use cases.

### Fine-tuning Benefits

- **Improved Task Performance**: Achieve better results on niche applications.
- **Enhanced Model Efficiency**: Tailored models require fewer computational resources for specific tasks.
- **Adaptability**: Teach the model new patterns or domain-specific knowledge.

---

## Dedicated AI Clusters

Dedicated AI clusters in OCI provide isolated GPU-based compute resources, ensuring high performance and security for customer-specific generative AI workloads. These clusters can be provisioned as either **shared** or **dedicated**:

- **Shared Clusters**: GPU resources are managed across multiple customers.
- **Dedicated Clusters**: Exclusively allocated GPUs for isolated generative AI tasks, offering enhanced performance and privacy.

The GPUs allocated for a customer’s generative AI tasks are isolated from others, ensuring no cross-interference in workloads.

---

## Getting Started with OCI Generative AI Service

1. **Access the API**: Use the single API provided by OCI to integrate various models into your applications.
2. **Select a Model**: Choose from a range of high-performing pretrained models to suit your specific use case.
3. **Fine-tune (Optional)**: If necessary, customize your chosen model with T-Few fine-tuning.
4. **Deploy on Dedicated AI Clusters**: Leverage GPU-based compute resources to efficiently handle training and inference workloads.
5. **Integrate**: Deploy your customized models into your applications, enabling enhanced AI-powered capabilities.
# OCI Generative AI Service - README

## Overview

The **OCI Generative AI Service** provides a fully managed platform to build generative AI applications with large language models (LLMs). The service offers a variety of pretrained and customizable LLMs accessible via a single API. With a flexible selection of models, fine-tuning capabilities, and dedicated AI clusters, the OCI Generative AI Service supports high-performance, scalable AI solutions tailored to your application's needs.

---

## Key Features

### 1. Tokens and Language Models
Language models interpret input as **tokens** rather than individual characters. Each token can be an entire word, a part of a word, or punctuation. Here’s how tokenization works:

- **Simple Words**: Generally map to a single token (e.g., "apple").
- **Complex Words**: Can map to multiple tokens (e.g., "friendship" maps to "friend" and "ship").
- **Token Counts**: Depends on text complexity. Simple text averages around one token per word, while complex text may require two to three tokens per word.

### 2. Model Options and Use Cases
OCI offers a variety of pretrained foundational models designed for specific use cases, such as:

| Model Name               | Provider   | Max Tokens | Parameters      | Use Cases                                          |
|--------------------------|------------|------------|-----------------|----------------------------------------------------|
| `command-r-plus`         | Cohere     | 128k       | -               | Chat, Q&A, information retrieval, sentiment analysis|
| `command-r-16k`          | Cohere     | 16k        | -               | Speed and cost-sensitive applications               |
| `llama-3.1-405b/70b`     | Meta       | 128k       | 405b/70b        | Large enterprise applications                       |
| `embed-english-v3.0`     | Cohere     | -          | -               | Semantic search in English                          |
| `embed-multilingual-v3.0`| Cohere     | -          | -               | Multilingual applications                           |

---

### 3. Fine-tuning and Customization
Fine-tuning enables users to tailor pretrained models to their specific needs. OCI provides **T-Few fine-tuning** (by Cohere), a quick and efficient method for model customization. Fine-tuning benefits include:

- **Enhanced Task Performance**: Fine-tuned models yield better results in niche applications.
- **Increased Efficiency**: Custom models often use fewer computational resources.
- **Domain-Specific Adaptability**: Models can learn new patterns or specialize in particular tasks.

---

## Model Control Parameters

### Maximum Output Tokens
Defines the maximum number of tokens a model generates in response to a prompt.

### Preamble Override
An initial contextual message that adjusts the model’s response style or behavior. By specifying a custom preamble, users can override the model’s default behavior. If not provided, the model uses a built-in preamble.

**Default Preamble for Cohere Models**:
> "You are Command. You are an extremely capable large language model built by Cohere. You are given instructions programmatically via an API that you follow to the best of your ability."

### Temperature
Controls the **randomness** of generated outputs:
- **Lower Temperature (0)**: Produces consistent, deterministic results suitable for tasks requiring precise answers, such as Q&A.
- **Higher Temperature (0.7+)**: Yields more creative outputs, but may introduce hallucinations or less accurate responses.

### Top-k Sampling
The model selects the next token from the **top k** tokens ranked by probability:
- **Top-k Example**: If set to 3, the model selects from the three most probable tokens, balancing creativity and relevance.

### Top-p Sampling (Nucleus Sampling)
Picks the next token from the **top cumulative probability** of tokens:
- **Top-p Example**: If set to 0.75, the model excludes the lowest 25% of probable tokens, limiting the output to the most relevant options.

### Frequency and Presence Penalties
These penalties help reduce repetitive responses in generated text:
- **Frequency Penalty**: Penalizes tokens based on the frequency of their occurrence.
- **Presence Penalty**: Penalizes tokens regardless of their frequency once they appear in the output, encouraging variety in the response.

---

## Dedicated AI Clusters

OCI’s dedicated AI clusters are GPU-based compute resources reserved for fine-tuning and inference tasks, ensuring high performance and data security:

- **Shared vs. Dedicated Clusters**: Choose shared clusters for standard workloads, while dedicated clusters offer isolated, optimized GPU resources for high-stakes, enterprise applications.

---

## Getting Started with OCI Generative AI Service

### Step 1: Access the API
OCI’s API provides a unified endpoint to manage multiple LLMs for diverse applications. Use it to integrate pretrained models directly into your applications.

### Step 2: Select a Model
Choose from a selection of models depending on the application's needs, from conversational models to embedding models for semantic search.

### Step 3: Fine-tune (Optional)
If necessary, fine-tune the chosen model to adapt it for domain-specific tasks, enhancing performance, efficiency, and relevance.

### Step 4: Deploy on Dedicated AI Clusters
Use OCI’s GPU-powered clusters to host fine-tuning and inference processes, ensuring optimal resource allocation and security.

### Step 5: Integrate
Integrate the model into your application to provide robust generative AI functionality, whether for chatbots, content generation, information retrieval, or multilingual support.

---

## Overview of Embeddings

Embeddings are numerical representations of text converted into sequences of numbers, allowing computers to understand relationships between different pieces of text. A text piece could be as short as a single word or as large as multiple paragraphs. Embeddings simplify the task of comparing or relating text, making them essential for tasks like search, classification, and natural language understanding.

---

## How Embeddings Work

Embeddings capture semantic information by representing each piece of text as a vector in a multi-dimensional space. The closer the vectors of two texts, the more semantically similar they are. This process is critical in various AI applications, as it allows for efficient text comparison and matching.

### Types of Embeddings

1. **Word Embeddings**
   - Word embeddings capture inherent properties of words, such as their semantics. For example, embeddings for words related to **age** and **size** would represent these properties as vectors in a space with axes aligned to these features.
   - Actual embeddings capture far more properties than just two, making them highly descriptive of the word’s meaning and usage.

2. **Sentence Embeddings**
   - Sentence embeddings represent entire sentences as vectors, enabling the model to capture the semantic similarity of phrases. Similar sentences have vectors close to each other in the vector space, while dissimilar sentences have vectors further apart.
   - Example: The sentence "canine companions say" would have an embedding closer to "woof" than "meow."

---

## Calculating Semantic Similarity

To determine similarity between embeddings, two primary techniques are used:

1. **Cosine Similarity**
   - Measures the cosine of the angle between two vectors. A smaller angle indicates higher similarity.

2. **Dot Product Similarity**
   - Measures the degree to which two vectors point in the same direction. Higher dot products signify greater similarity.

For example, the embedding vector of "Puppy" will be more similar to "Dog" than "Lion." This method helps group embeddings into clusters based on semantic similarity, such as animals, fruits, or places.

---

## Embedding Use Case in Generative AI

In a generative AI application, embeddings play a vital role in information retrieval and answering user questions.

1. **User Query Encoding**: The user's question is encoded as an embedding vector.
2. **Vector Database Search**: This embedding is used to search a vector database, retrieving relevant private content or documents based on semantic similarity.
3. **LLM Contextualization**: The content from the vector database is then sent to the LLM, enabling it to answer the user's question more accurately by combining the retrieved information with general knowledge.

---

## Embedding Models in OCI Generative AI

OCI offers multiple embedding models tailored to different needs, such as handling English or multilingual text. Each model produces vectors of specific dimensionalities, optimized for different tasks.

### Model Options

| Model Name                  | Provider   | Dimension | Max Tokens | Description                                                   |
|-----------------------------|------------|-----------|------------|---------------------------------------------------------------|
| `embed-english-v3.0`        | Cohere     | 1024      | 512        | Converts English text into high-dimensional vector embeddings. |
| `embed-multilingual-v3.0`   | Cohere     | 1024      | 512        | Multilingual embedding for 100+ languages.                     |
| `embed-english-light-v3.0`  | Cohere     | 384       | 512        | Faster, smaller English embedding model.                       |
| `embed-multilingual-light-v3.0` | Cohere | 384       | 512        | Faster, smaller multilingual embedding model.                  |
| `embed-english-light-v2.0`  | Cohere     | 1024      | 512        | Previous generation English embedding model.                   |

---

### Embedding Model Descriptions

- **embed-english-v3.0**: Creates 1024-dimensional vectors for English text, suitable for high-quality semantic search and similarity tasks.
- **embed-multilingual-v3.0**: Supports over 100 languages, producing 1024-dimensional vectors for diverse linguistic applications.
- **embed-english-light-v3.0**: A lightweight, efficient model for English text, generating 384-dimensional vectors for faster processing.
- **embed-multilingual-light-v3.0**: A multilingual version of the lightweight model, suitable for applications requiring lower latency and 384-dimensional vector outputs.
- **embed-english-light-v2.0**: The previous version of the English model, producing 1024-dimensional vectors.

---

## Practical Applications

### 1. Semantic Search
Use embeddings to enhance search functionality by enabling it to understand and match content based on meaning rather than exact keyword matches.

### 2. Information Retrieval
Retrieve relevant documents or responses from a database by comparing the embedding of a query to stored embeddings.

### 3. Multilingual Support
With multilingual embeddings, support cross-language information retrieval, clustering, and classification tasks.

Advanced Prompting Strategies
Chain-of-Thought - provide examples in a prompt is to show responses that include a
reasoning step
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can
has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis
balls. 5 + 6 = 11. The answer is 11.
[Wei et al, 2022]
Zero Shot Chain-of-Thought - apply chain-of-thought prompting without providing
examples
Q: A juggler can juggle 16 balls. Half of the balls are golf balls,
and half of the golf balls are blue. How many blue golf balls are
there?
A: Let's think step by step.""»
[Kojima et al, 2022]

# Dedicated AI Clusters Sizing and Pricing - OCI Generative AI Service

## Overview

The **OCI Generative AI Service** offers Dedicated AI Clusters tailored for fine-tuning and hosting language models. These clusters are provisioned as units, and each unit type varies in size and capacity to support specific models and tasks. This document provides guidelines on choosing and pricing dedicated AI clusters for different workloads.

---

## Dedicated AI Cluster Unit Types

There are multiple types of dedicated AI cluster units, each suited for different tasks like text generation, summarization, or embeddings. Below is a list of available unit types:

| Unit Size           | Model Type                 | Description                                           | Limit Name                     |
|---------------------|----------------------------|-------------------------------------------------------|--------------------------------|
| **Large Cohere**    | Text Generation            | Fine-tuning/hosting **cohere.command** model          | `dedicated-unit-large-cohere-count` |
| **Small Cohere**    | Text Generation            | Fine-tuning/hosting **cohere.command-light** model    | `dedicated-unit-small-cohere-count` |
| **Embed Cohere**    | Embedding                  | Hosting **cohere.embed** models                       | `dedicated-unit-embed-cohere-count` |
| **Llama2-70**       | Text Generation, Chat      | Hosting **llama2_70b-chat** models                    | `dedicated-unit-llama2-70-count` |

---

## Cluster Sizing Guidelines

### Fine-Tuning Dedicated AI Cluster
Fine-tuning a model requires more compute resources compared to hosting. Each fine-tuning cluster typically uses **two units**.

- **Large Cohere**: 2 units for fine-tuning **cohere.command**
- **Small Cohere**: 2 units for fine-tuning **cohere.command-light**

> **Example**: To fine-tune a **cohere.command** model, you would need a cluster with **two Large Cohere units**.

### Hosting Dedicated AI Cluster
Hosting a model requires fewer resources. Each hosting cluster typically requires **one unit**.

- **Large Cohere**: 1 unit for hosting **cohere.command**
- **Small Cohere**: 1 unit for hosting **cohere.command-light**
- **Embed Cohere**: 1 unit for hosting **cohere.embed**
- **Llama2-70**: 1 unit for hosting **llama2_70b-chat**

> **Example**: To host a fine-tuned **cohere.command** model, you would need a **one Large Cohere unit**.

---

## Cluster Capacity and Flexibility

- **Fine-Tuning Clusters**: Each fine-tuning cluster can be reused to fine-tune multiple models.
- **Hosting Clusters**:
  - Can host up to **50 fine-tuned models**.
  - Can support up to **50 endpoints**, each pointing to a different model hosted on the same cluster.

---

## Example Pricing Breakdown

Here’s an example of estimating costs for a user (e.g., Bob) who fine-tunes and hosts a **Cohere command** model each week.

### Fine-Tuning Costs

- **Fine-Tuning Cluster Requirements**: 2 Large Cohere units
- **Fine-Tuning Time per Job**: 5 hours
- **Fine-Tuning Jobs per Month**: 4 (one per week)
- **Unit-Hours for Fine-Tuning**:
  - 5 hours per job × 2 units = **10 unit-hours per job**
  - 10 unit-hours per job × 4 jobs = **40 unit-hours per month**

#### Fine-Tuning Cost Calculation
\[
\text{Fine-Tuning Cost per Month} = 40 \text{ unit-hours} \times \text{Large Cohere unit price per hour}
\]

### Hosting Costs

- **Hosting Cluster Requirements**: 1 Large Cohere unit
- **Minimum Hosting Commitment**: 744 unit-hours (based on 24/7 availability)

#### Hosting Cost Calculation
\[
\text{Hosting Cost per Month} = 744 \text{ unit-hours} \times \text{Large Cohere unit price per hour}
\]

### Total Monthly Cost
\[
\text{Total Monthly Cost} = (\text{Fine-Tuning Cost per Month} + \text{Hosting Cost per Month})
\]
\[
= (40 + 744) \text{ unit-hours} \times \text{Large Cohere unit price per hour}
\]

---

## Summary

- **Fine-Tuning** requires a cluster with **2 units** per model.
- **Hosting** requires a cluster with **1 unit** per model.
- **Minimum Commitments**:
  - Fine-Tuning: 1 unit-hour per fine-tuning job.
  - Hosting: 744 unit-hours per cluster per month.
  
This pricing model enables businesses to estimate the cost based on their specific model fine-tuning and hosting requirements. For the latest pricing, consult the [OCI Generative AI pricing documentation](https://docs.oracle.com/en-us/iaas/generative-ai/docs/).
# Fine-Tuning Configuration for OCI Generative AI Service

Fine-tuning a language model enables it to specialize in specific tasks by adjusting model parameters on a custom dataset. The OCI Generative AI Service offers flexible fine-tuning configurations, including the **Vanilla** and **T-Few** training methods. This document provides an overview of fine-tuning methods, hyperparameters, and evaluation metrics.

---

## Training Methods

1. **Vanilla Fine-Tuning**: Traditional approach that updates a large portion or all layers in the model, typically resulting in longer training times and higher inference costs.
2. **T-Few Fine-Tuning**: An efficient fine-tuning technique that selectively updates only a small fraction of layers, significantly reducing training time and computational cost.

---

## Hyperparameter Configuration

Fine-tuning models involve setting specific hyperparameters to control the training process. Below are the hyperparameters available for configuration and their default values.

### Common Hyperparameters

| Hyperparameter                | Description                                                                                          | Default Value              |
|-------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------|
| **Total Training Epochs**     | Number of iterations through the entire training dataset.                                             | 3                           |
| **Learning Rate**             | The rate at which model parameters are updated after each batch.                                      | 0.1 (T-Few), 0.01 (Vanilla) |
| **Training Batch Size**       | Number of samples processed before updating model parameters.                                         | 8 for `cohere.command`, 8–16 for `cohere.command-light` |
| **Early Stopping Threshold**  | Minimum improvement in loss required to prevent early termination of training.                        | 0.01                        |
| **Early Stopping Patience**   | Tolerance for stagnation in the loss metric before stopping training.                                | 6                           |
| **Log Model Metrics Interval**| Frequency for logging model metrics. Initially logs every step for the first 20 steps, then uses this parameter for subsequent steps. | 10                          |

### Vanilla-Specific Parameter

- **Number of Last Layers**: Specifies the number of layers to update during fine-tuning.

---

## Fine-Tuning Parameters for T-Few

T-Few uses selective parameter updates focused on added transformer layers. T-Few’s hyperparameters allow efficient, targeted fine-tuning, making it suitable for large-scale models where training all layers would be computationally expensive.

### T-Few Hyperparameters

| Hyperparameter                | Description                                                                                          | Default Value              |
|-------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------|
| **Total Training Epochs**     | Number of iterations through the entire training dataset.                                             | 3                           |
| **Batch Size**                | Number of samples processed before updating model parameters.                                         | Default 8–16                |
| **Learning Rate**             | Learning rate for updating T-Few layers.                                                              | 0.1                         |
| **Early Stopping Threshold**  | Minimum improvement in loss required to continue training.                                            | 0.01                        |
| **Early Stopping Patience**   | Tolerance for stagnation in the loss metric before stopping training.                                | 6                           |
| **Log Model Metrics Interval**| Frequency of logging model metrics.                                                                   | 10                          |

---

## Understanding Fine-Tuning Results

Two primary metrics are used to evaluate the performance of a fine-tuned model:

### 1. Accuracy
- **Definition**: Accuracy is the ratio of correct predictions to the total predictions made during evaluation.
- **Evaluation**: To assess accuracy, the model is tasked with predicting specific words in the user-provided dataset. Higher accuracy indicates the model is performing well on the task.

### 2. Loss
- **Definition**: Loss measures how far off the model’s predictions are from the correct answer. It helps quantify the degree of incorrectness in predictions.
- **Evaluation**: Lower loss indicates that the model’s predictions are closer to the correct answer, showing improvement as the model trains.
- **Goal**: During fine-tuning, loss should decrease over epochs, reflecting that the model is learning effectively from the training data.

---
# Security and Privacy in OCI Generative AI Service

The **OCI Generative AI Service** prioritizes security and privacy, ensuring that customer data and models are isolated and protected throughout the generative AI lifecycle. Dedicated GPU clusters and secure network infrastructure provide strong isolation, enabling customers to fine-tune and host models with confidence.

---

## Key Security Features

### 1. Dedicated GPU and RDMA Network

- **GPU Isolation**: GPUs assigned to a customer’s generative AI tasks are isolated from those used by other customers, preventing cross-customer access or data leakage.
- **Dedicated RDMA Network**: GPU clusters operate within a secure **Remote Direct Memory Access (RDMA) network**, ensuring high performance while maintaining strong data isolation.

---

## Infrastructure View

The infrastructure design ensures both physical and logical isolation of resources:

- **Physical View**: Each dedicated AI cluster is composed of GPUs allocated specifically for a single customer’s workloads.
- **Logical View**: The dedicated AI cluster and its GPUs are organized within a secure RDMA network, ensuring that the customer’s fine-tuned models are hosted in a secure, isolated environment.

---

## Tenancy Model and Endpoints

OCI’s tenancy model enforces strict data privacy and security measures:

- **Single-Customer GPU Cluster**: Each dedicated GPU cluster serves only the fine-tuned models and endpoints of a single customer, isolating their data and workloads from others.
- **Endpoint Sharing**: A dedicated cluster can host multiple model endpoints, such as the base model endpoint and several fine-tuned custom model endpoints. This sharing enables efficient GPU utilization within the cluster without compromising isolation.

---

### Endpoint Organization in Dedicated Clusters

A customer’s dedicated cluster can support multiple endpoints, including:

- **Base Model Endpoint**: Provides access to the foundational model.
- **Custom Model Endpoints (A, B, C)**: These represent different fine-tuned models within the same cluster, each accessible only to the customer’s application.

---

## Customer Data and Model Isolation

Customer data and model information are isolated within each customer’s OCI tenancy, preventing unauthorized access:

- **Restricted Data Access**: Data and models are accessible only within the customer’s specific OCI tenancy, and customer-specific applications can exclusively access custom models.
- **Tenancy Isolation**: Multiple customer tenancies (e.g., Customer 1 and Customer 2) run independently, each with its own applications and models isolated from one another.

---

## Leveraging OCI Security Services

The OCI Generative AI Service integrates with several OCI security services to enhance data protection:

- **OCI Identity and Access Management (IAM)**: Used for authentication and authorization, IAM ensures that only authorized users and applications can access the AI models.
- **OCI Key Management Service**: Securely stores model encryption keys, safeguarding the base model weights and other sensitive data.
- **OCI Object Storage**: Fine-tuned model weights and related data are stored in encrypted OCI Object Storage buckets by default, adding an additional layer of security.

---

### Security Architecture Overview

1. **Customer Tenancy**: Each customer has a secure tenancy within OCI, where only authorized applications can access models.
2. **IAM for Authentication and Authorization**: Controls access to models and endpoints.
3. **Key Management Service**: Manages and securely stores encryption keys.
4. **Object Storage for Model Weights**: Stores both base model weights and fine-tuned model weights in encrypted storage.

---

## Summary

The OCI Generative AI Service incorporates robust security measures to protect customer data and model assets. With dedicated GPU clusters, an RDMA network, and integration with OCI security services, customers can securely fine-tune and host AI models. OCI’s infrastructure and isolation policies ensure that data remains private and accessible only within each customer’s specific OCI tenancy.

For more details, refer to the [OCI Generative AI Security documentation](https://docs.oracle.com/en-us/iaas/generative-ai/docs/).
## Summary of the Fine-Tuning Workflow

1. **Choose Fine-Tuning Method**: Select between Vanilla (all layers) and T-Few (efficient layer selection).
2. **Configure Hyperparameters**: Set values for epochs, learning rate, batch size, and early stopping as needed.
3. **Run Fine-Tuning**: Start the fine-tuning process on a dedicated AI cluster, monitoring the accuracy and loss metrics to evaluate model improvement.
4. **Evaluate Results**: Monitor accuracy and loss to determine the effectiveness of the fine-tuning, aiming for high accuracy and low loss.

This configuration enables efficient model customization for specific tasks.

