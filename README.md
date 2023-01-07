
## The Impact of Symbolic Representations on In-context Learning for Few-shot Reasoning

<img  src="/assets/lmlp.png" width="550">

This is the official demo code for our [The Impact of Symbolic Representations on In-context Learning for Few-shot Reasoning](https://openreview.net/forum?id=qLgQpeQX3x1) paper. The code demonstrates how Large Language Models, such as GPT-3, can generate reasoning provenance using in-context learning. 
### Requirements
```pip -r install requirements.txt```

## Running Code

The dataset structure of CLUTRR-LP:

- data/clutrr/example_all.json: The logic rules.
- data/clutrr/example_test.json: The query
- data/clutrr/rules_all.json: The facts

The dataset structure of Countries-LP, task Si:

- data/countries/avaliable_examples_ri.json: The logic rules.
- data/countries/test_samples.json: The query
- data/countries/avaliable_rules_ri.json: The facts

Run experiment of LMLP for the Countries-LP:

```Python src/countries.py (The logic rules and facts are set in the code)```

Run experiment of LMLP for the CLUTRR-LP:

```Python src/clutrr.py --num_rule 1 --rule_path [The facts path] --example_path [The logic rules] --test_path [The query]```

Run experiment of CoT for the CLUTRR-LP:

```Python src/clutrr_cot.py```


If you find this work helpful for your research, please consider citing:
```bibtex
@inproceedings{
  zhang2022the,
  title={The Impact of Symbolic Representations on In-context Learning for Few-shot Reasoning},
  author={Hanlin Zhang and YiFan Zhang and Li Erran Li and Eric Xing},
  booktitle={NeurIPS 2022 Workshop on Neuro Causal and Symbolic AI (nCSI)},
  year={2022},
  url={https://openreview.net/forum?id=qLgQpeQX3x1}
}
```