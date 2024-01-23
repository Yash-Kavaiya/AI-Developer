# Introduction of MLOps

## Software Development Life Cycle

Software Development Life Cycle, is a structured and systematic process used by software developers and project managers to plan, design, build, test, deploy, and maintain high-quality software systems. The goal of SDLC is to produce software that meets or exceeds customer expectations, is delivered on time and within budget, and is easy to maintain and update. The life cycle typically consists of several phases, each with its own set of activities and deliverables. Here is an overview of the typical SDLC phases:

1. **Planning:**
   - **Objective:** Define the project scope, requirements, budget, and timeline.
   - **Activities:**
     - Conduct feasibility studies.
     - Define project goals and objectives.
     - Create a project plan and schedule.
     - Identify risks and mitigation strategies.
     - Obtain approval from stakeholders.

2. **Analysis:**
   - **Objective:** Gather detailed requirements and analyze them for feasibility.
   - **Activities:**
     - Collect and document functional and non-functional requirements.
     - Conduct interviews, surveys, and workshops with stakeholders.
     - Analyze existing systems (if applicable).
     - Create use cases, user stories, and system specifications.

3. **Design:**
   - **Objective:** Create a blueprint for the software architecture.
   - **Activities:**
     - Develop system architecture and design.
     - Define data structures, algorithms, and interfaces.
     - Create detailed technical specifications.
     - Design user interfaces.
     - Plan for security and performance considerations.

4. **Implementation (Coding):**
   - **Objective:** Transform the design into actual code.
   - **Activities:**
     - Write code according to the specifications.
     - Conduct unit testing to ensure individual components work as intended.
     - Implement version control and manage source code.

5. **Testing:**
   - **Objective:** Verify that the software meets the specified requirements and is free of defects.
   - **Activities:**
     - Conduct various levels of testing (unit, integration, system, acceptance).
     - Identify and fix bugs and issues.
     - Perform regression testing to ensure changes do not negatively impact existing functionality.

6. **Deployment:**
   - **Objective:** Release the software for general use.
   - **Activities:**
     - Develop installation procedures.
     - Deploy the software to production environments.
     - Monitor and troubleshoot any issues during deployment.

7. **Maintenance and Support:**
   - **Objective:** Ensure the ongoing functionality and performance of the software.
   - **Activities:**
     - Provide user support.
     - Address reported issues and bugs.
     - Implement updates and patches as needed.
     - Enhance or extend functionality based on user feedback.

8. **Retirement/Phase-out:**
   - **Objective:** Decommission the software when it reaches the end of its useful life.
   - **Activities:**
     - Archive data and documentation.
     - Notify users about the retirement.
     - Plan for the migration to a new system if necessary.

| Aspect                   | Waterfall                        | Agile                            | Spiral                           | V-Model                          |
|--------------------------|----------------------------------|----------------------------------|----------------------------------|----------------------------------|
| **Development Approach** | Sequential                      | Iterative and Incremental        | Iterative and Incremental        | Sequential and Parallel          |
| **Flexibility**           | Low                              | High                             | Moderate                         | Moderate                         |
| **Phases**                | Sequential (Linear)             | Iterative (Cyclic)               | Iterative (Cyclic)               | Parallel (V-Shaped)              |
| **Feedback**              | Limited until the end            | Continuous throughout            | Ongoing throughout                | At each stage                    |
| **Client Involvement**    | In the beginning and the end     | Throughout the development      | Ongoing throughout                | At each stage                    |
| **Risk Management**       | Late identification of issues    | Early and continuous assessment  | Ongoing and adaptive              | At each stage                    |
| **Delivery Time**         | Longer delivery cycles           | Shorter delivery cycles           | Moderate delivery cycles         | Moderate delivery cycles         |
| **Change Management**     | Difficult to accommodate changes | Embraces changes easily           | Can accommodate changes easily   | Some flexibility                |
| **Testing**               | At the end of the development    | Concurrent with development      | Ongoing throughout                | Parallel with development        |
| **Documentation**         | Comprehensive documentation      | Minimal documentation             | Ongoing and updated documentation | Detailed documentation           |


## Differences between MLOps and DevOps.

| Aspect                  | MLOps                            | DevOps                           |
|-------------------------|----------------------------------|----------------------------------|
| **Focus**               | Machine Learning (ML) model deployment, monitoring, and management. | Software development, deployment, and operations. |
| **Primary Goal**        | Efficient deployment and management of ML models in production. | Streamlining and automating the software development lifecycle. |
| **Nature of Artifacts** | ML models, datasets, and experiment tracking. | Code, configuration files, and application artifacts. |
| **Integration**         | Integration of ML workflows and model training into the deployment pipeline. | Integration of development, testing, and deployment processes. |
| **Iterations**          | Involves multiple iterations of model training and deployment. | Focuses on continuous integration, continuous deployment, and continuous testing. |
| **Testing**             | Involves model testing, validation, and monitoring for accuracy. | Emphasizes code testing, unit testing, and system testing. |
| **Tools**               | ML-specific tools (e.g., TensorFlow, PyTorch) and ML frameworks. | General-purpose tools (e.g., Jenkins, Docker) and infrastructure automation tools. |
| **Metrics**             | Involves ML-specific metrics like accuracy, precision, and recall. | Focuses on metrics related to software development and deployment speed, reliability, and efficiency. |
| **Collaboration**       | Collaboration between data scientists, ML engineers, and operations teams. | Collaboration between development, testing, and operations teams. |
| **Environment**         | Involves specialized ML environments and infrastructure. | General-purpose development and deployment environments. |
| **Challenges**          | Challenges related to model versioning, data drift, and model interpretability. | Challenges related to collaboration, automation, and orchestration. |
| **Examples**            | Kubeflow, MLflow, TensorFlow Extended (TFX). | Jenkins, Docker, Ansible, Kubernetes. |

