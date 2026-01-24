<p align="center">
  <img src="src/hype-edit-1-cover.jpg" alt="HYPE-EDIT-1 cover" style="max-width: 672px; width: 100%; height: auto;" />
</p>

# HYPE-EDIT-1

Benchmark for measuring reliability in frontier image editing models

There is a big gulf between the hype marketing promoting generative AI models and their real world performance, and we wanted to take some time to measure this quantitatively. We are pleased to release our first public benchmark, called HYPE-EDIT-1, to measure the image editing performance of leading models on their reliability and effective cost on real world test cases based on actual tasks.

Current generative AI benchmarks and the marketing of the models by both influencers and company employees tends to focus on cherrypicked best examples of the model's capability. However, for real world business use cases, it needs to perform reliably well, not just fantastically from time to time.

Our benchmark focuses on real world image editing tasks in marketing/design context, but is evaluated differently to most benchmarks:

1. We run a given task through the model 10 times and judge each candidate. This gives us a success rate over repeated attempts.
2. We use the per-candidate pass rate to estimate expected attempts (capped at 4) and effective cost per success. This highlights the tradeoff between price per image and reliability.

## Why is this important?

We think this benchmark is important because it highlights the gap between the current leading model's potential capability (as highlighted in their marketing materials) and the actual lived experience of users trying to use the models at scale to replace, evolve or change their current workflows.

We also are then able to provide a "cost of usage" calculation that is more useful and nuanced than just the cost per image maths that models are currently compared on. Whether an image model is better for you versus another should be based not just on the cost per image but also on the number of attempts to get a successful image.

Finally, we hope that this benchmark inspires teams to make models that are more reliable.

## What's in the benchmark

- There are 100 carefully curated image tasks all using reference imagery. Some are single image editing, some are multi-image editing, and none of them are "trick" question tasks. Real world tasks are not "count how many letters are in strawberry" or "visualise this as a 4D hypercube".
- Each task is run 10x per model, with a judge VLM to decide if the task was completed successfully or not based on a scoring threshold.
- This then gives us four main metrics:
  1. Pass rate (P@1, image-level reliability), which is the % of the 1000 images per model that are successful.
  2. Pass@10 (P@10, task-level reliability), which is the % of tasks with at least one PASS.
  3. Pass@4 (P@4), the % of tasks with at least one PASS within 4 attempts.
  4. Effective cost per success, which accounts for repeat attempts to hit a PASS (see more below).

## Calculating Effective Cost per Success

We cap attempts at 4. For each task, we estimate per-candidate pass rate p from the 10 candidates and compute the expected attempts to get a PASS within 4 tries:

$$
E = \frac{1 - (1 - p)^4}{p}
$$

(with E = 4 when p = 0), and the success probability within 4 tries:

$$
S = 1 - (1 - p)^4
$$

We aggregate across tasks and compute Pass@4 and expected attempts:

$$
C_{success} = \frac{E \cdot C_{attempt}}{P@4}
$$

where C_attempt = model cost + review cost. This favors models that are reliable, not just cheap per image.

## Why are models unreliable?

There are three main hypotheses we have for why current models are quite unreliable.

1. The models are still not good enough. We need larger datasets, more training and better architectures. This is the 'bitter pill' explanation.
2. There is some infrastructure optimisation happening behind the scenes. We've suspected for a while that a "model" is not uniformly served at all times, and under heavy load or across different infrastructures, there is actually a difference happening in inference. This is hard to prove for private models, but even for open weights models, a lot of optimisation happens when infra clouds serve the models that affects reliability.
3. There hasn't been a benchmark highlighting this until now. This is a similar argument that OpenAI made about hallucination being a property of bad benchmark science that penalised correctly saying I don't know instead of hedging your bets and trying to produce an answer that somewhere contained what the examiner was looking for.

## How to run the benchmark

The benchmark consists of two datasets, which is becoming fairly standard in the industry now:

1. Public dataset - this is available on Github. 50 test cases.
2. Private dataset - this is administered by Sourceful on models through public APIs to avoid training on test. 50 test cases.

Combined results are the union of public + private and are used for reporting (see `RESULTS.md`).

Each test case specifies one or more filenames used as reference input images.

Each image is available on https://cdn.sourceful.com/research/benchmarks/hype-edit-1/tasks/{task_id}/{filename}

The dataset consists of pairs of Reference images and an Instruction prompt. We provide a reference implementation that uses Gemini 3 Flash (gemini-3-flash-preview) that works well, as well as a web UI if you wish to do human review (model is anonymous to the reviewer).


## Intended Impact

HYPE-EDIT-1 is a narrowly scoped benchmark that helps model researchers and users in the target audience (marketers, designers, agencies in the digital and physical commerce market) understand quantitatively more about model performance.

We hope to see the scores in this benchmark improve over time.

## Citation

If you use this benchmark in academic or product work, please cite:

```bibtex
@misc{hype-edit-1,
  title = {HYPE-EDIT-1: An Effective-Cost and Reliability
Benchmark for Image Editing},
  year = {2026},
  howpublished = {arxiv},
  url = {https://github.com/sourceful/hype-edit-1-benchmark}
}
```

## License

- Code in this repository is licensed under the MIT License (see `LICENSE`).
- The public task data including reference imagery in `src/tasks/public.json` is licensed under CC BY 4.0 (see `LICENSE-DATA`).

