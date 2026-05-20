<div align="center">

# Stitched Value Model for Diffusion Alignment

[Hyojun Go](https://gohyojun15.github.io/)<sup>1</sup> &nbsp;·&nbsp;
[Hyungjin Chung](https://hyungjin-chung.github.io/) &nbsp;·&nbsp;
[Prune Truong](https://prunetruong.com/)<sup>2</sup> &nbsp;·&nbsp;
[Goutam Bhat](https://goutamgmb.github.io/)<sup>2</sup> &nbsp;·&nbsp;
[Li Mi](https://limirs.github.io/)<sup>1</sup> &nbsp;·&nbsp;
[Zhaochong An](https://zhaochongan.github.io/)<sup>3</sup>
<br>
[Zixiang Zhao](https://zhaozixiang1228.github.io/)<sup>1</sup> &nbsp;·&nbsp;
[Dominik Narnhofer](https://baug.ethz.ch/departement/personen/mitarbeiter/personen-detail.MzM5ODU5.TGlzdC82NzksLTU1NTc1NDEwMQ==.html)<sup>1</sup>
<br>
[Serge Belongie](https://sergebelongie.github.io/)<sup>3</sup> &nbsp;·&nbsp;
[Federico Tombari](https://federicotombari.github.io/)<sup>2</sup> &nbsp;·&nbsp;
[Konrad Schindler](https://prs.igp.ethz.ch/group/people/person-detail.schindler.html)<sup>1</sup>

<sup>1</sup>ETH Zürich &nbsp;·&nbsp;
<sup>2</sup>Google &nbsp;·&nbsp;
<sup>3</sup>University of Copenhagen

[**Project page**](https://gohyojun15.github.io/StitchVM/) &nbsp;|&nbsp;
[**Paper (PDF)**](https://gohyojun15.github.io/StitchVM/static/stitchvm.pdf) &nbsp;|&nbsp;
[**arXiv**](https://arxiv.org/abs/2605.19804)

</div>

---

<p align="center">
  <img src="https://gohyojun15.github.io/StitchVM/static/images/fig_overview-1.png" alt="StitchVM method overview" width="92%">
</p>

**StitchVM** turns any pretrained pixel reward model (CLIP, HPSv2, Aesthetic Predictor, …) into a **value model that scores noisy diffusion latents directly**, at small compute cost. Drop the resulting value model into any diffusion-alignment recipe (DPS, FK steering, DRaFT, DiffusionNFT, …) and each gets cheaper *and* often better at the same time.

See the **[project page](https://gohyojun15.github.io/StitchVM/)** for the full story, figures, and results.

---

## 🚧 Code release

> The code release is **in preparation**. Star or watch this repository to be notified.

## Citation

```bibtex
@article{go2026stitchvm,
    title   = {Stitched Value Model for Diffusion Alignment},
    author  = {Go, Hyojun and Chung, Hyungjin and Truong, Prune and Bhat, Goutam
               and Mi, Li and An, Zhaochong and Zhao, Zixiang and Narnhofer, Dominik
               and Belongie, Serge and Tombari, Federico and Schindler, Konrad},
    journal = {arXiv preprint arXiv:2605.19804},
    year    = {2026}
}
```

## Contact

For questions about the method or paper, please open a GitHub issue or contact [Hyojun Go](mailto:gohyojun15@gmail.com).
