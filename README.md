# A Weakly Supervised Knowledge Distillation Technique For Efficient Lung Nodule Segmentation (under review in International Joint Conference In Neural Networks(IJCNN-2026))

Abstract—Accurate pixel-level annotation of lung nodules in CT scans is difficult and expensive, limiting the applicability of fully supervised segmentation approaches. To address this challenge, this study presents a weakly supervised lung nodule segmentation framework that leverages coarse bounding-box annotations and knowledge distillation. A teacher–studentarchitecture is adopted, where a high-capacity Swin-UNet model with 45M parameters is first trained using coarse labels and subsequently used to generate pseudo labels. These pseudo-labels are then used to train a compact, modified U-Net model with only 10M parameters, and final evaluation is performed on finely annotated ground truth data. Model efficiency is further enhanced through the use of depthwise separable residual blocks, reducing the parameter count by approximately 50% compared to a standard U-Net. Instead of traditional logit-based distillation, feature-level knowledge transfer is explored, with low-level feature distillation yielding superior segmentation performance (Dice 50.03%, IoU 33.36%). Moreover, the integration of an Atrous Spatial Pyramid Pooling (ASPP) module enables effective multiscale context modeling, resulting in improved accuracy (Dice 51.19%, IoU 34.40%) and a reduced HD95 of 15.08, comparable to the teacher model’s HD95 of 13.36. The proposed approach achieves competitive segmentation performance while significantly reducing model parameters and computational cost compared to baseline methods.

Methodology
<img width="1391" height="757" alt="image" src="https://github.com/user-attachments/assets/a733ea28-1162-40b3-b90c-d64a0c206912" />


<h2>Results</h2>

<table>
  <thead>
    <tr>
      <th>Teacher Model</th>
      <th>Student Model</th>
      <th>Acc (%) ↑</th>
      <th>Pre ↑</th>
      <th>Rec (%) ↑</th>
      <th>Spec (%) ↑</th>
      <th>Harmonic (%) ↑</th>
      <th>Dice (%) ↑</th>
      <th>IoU (%) ↑</th>
      <th>HD95 ↓</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td colspan="10"><b>Baseline Teacher Models (Standalone Performance)</b></td>
    </tr>
    <tr>
      <td>U-Net</td><td>-</td><td><b>99.10</b></td><td>31.45</td><td>67.76</td><td><b>99.16</b></td><td>80.50</td><td>30.02</td><td>23.24</td><td><b>12.52</b></td>
    </tr>
    <tr>
      <td>U-Net++</td><td>-</td><td>98.43</td><td><b>33.87</b></td><td>94.85</td><td>98.46</td><td>96.62</td><td><b>49.91</b></td><td><b>33.26</b></td><td>19.76</td>
    </tr>
    <tr>
      <td>ResU-Net</td><td>-</td><td>98.37</td><td>33.22</td><td>96.30</td><td>98.39</td><td>97.33</td><td>49.39</td><td>32.80</td><td>18.75</td>
    </tr>
    <tr>
      <td>SwinUNet</td><td>-</td><td>98.35</td><td>33.10</td><td><b>98.54</b></td><td>98.34</td><td><b>98.44</b></td><td>49.55</td><td>32.93</td><td>13.36</td>
    </tr>
    <tr>
      <td>Attention U-Net</td><td>-</td><td>98.33</td><td>32.56</td><td>95.25</td><td>98.36</td><td>96.78</td><td>48.53</td><td>32.04</td><td>17.78</td>
    </tr>
    <tr>
      <td colspan="10"><b>Baseline Teacher–Student KD Models</b></td>
    </tr>
    <tr>
      <td>U-Net</td><td>U-NetDw</td><td><b>99.22</b></td><td>32.55</td><td>55.53</td><td><b>99.35</b></td><td>71.24</td><td>28.14</td><td>20.27</td><td>21.67</td>
    </tr>
    <tr>
      <td colspan="10"><b>Proposed Distillation Methods (Ours)</b></td>
    </tr>
    <tr>
      <td>SwinUNet</td><td>U-NetDw (KD)</td><td>98.55</td><td>34.98</td><td>87.83</td><td>98.64</td><td>92.92</td><td>50.03</td><td>33.36</td><td>26.26</td>
    </tr>
    <tr>
      <td>SwinUNet</td><td>U-NetDw (MKD)</td><td>98.52</td><td><b>35.15</b></td><td><b>94.15</b></td><td>98.56</td><td><b>96.30</b></td><td><b>51.19</b></td><td><b>34.40</b></td><td><b>15.08</b></td>
    </tr>
  </tbody>
</table>

<p>
<b>U-NetDw</b> = Depthwise Separable U-Net |
<b>KD</b> = Knowledge Distillation |
<b>MKD</b> = Multi-Scale Knowledge Distillation<br>
<b>Harmonic</b> = 2 × (Rec × Spec) / (Rec + Spec)
</p>
