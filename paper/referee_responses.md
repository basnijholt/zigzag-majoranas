Dear Daniel Ucko,

We would like to resubmit our revised manuscript where we address the referees' remarks.
We also address the referees' questions, point by point, below.
Finally, we also list the changes we have made to the manuscript.

Sincerely,  
The authors

## Report of Referee A -- LD17005/Laeven
> The manuscript discusses numerical calculations related to how to improve the energy gap in 2D planar Josephson junctions.
> The manuscript considers 2D SNS junctions within clean-limit BdG tight-binding model, used also in previous studies of the same system. Specific N region geometry is considered, and found to improve the energy gap. The argument why this occurs appears clear, and the results valid. Improving energy gaps likely is useful, but it's less clear this is the main/only problem faced in the experiments.
> The effect of long trajectories in ballistic systems is itself not surprising or new, but optimizing the geometry for this in the specific context of Majorana realizations with planar SNS was apparently not considered before. The closer study of this is the main new contribution here.

We thank the referee for the overall positive evaluation.
However, the referee's description seems to imply that the idea of optimizing the geometry of Josephson junction was known in other contexts.
To the best of our knowledge this is not the case, and this method was not considered in any context, not just in the Majorana devices.

> However, the manuscript appears better suited for publication in PRB. Although optimizing parameters in a previously discussed model is interesting and probably useful for future experiments of this type, it does not appear a significant advance of wide interest to a broad audience.

Our work reports an idea of adjusting the geometry as a way of influencing a phenomenon that is known for decades.
That this was overlooked for so long, in a very mature field, is in our opinion what makes it exciting and interesting.
Further, we predict a change in the device properties by more than an order of magnitude.
This is a substantial advance in the field of Majorana devices and therefore, makes our work satisfy the PRL criterion of importance.
The high degree of interest within the community is indicated by the ongoing experimental efforts of four independent experimental research groups, for example see [this March Meeting abstract](http://meetings.aps.org/Meeting/MAR20/Session/L60.6).

## Report of Referee B -- LD17005/Laeven
> The authors consider a planar Josephson junction in a zigzag geometry and propose it as an improved realization of a one-dimensional topological superconductor. As they show using numerical simulations, the zigzag geometry increases the induced superconducting gap and thereby decreases the spatial size of the Majorana wave functions, making them more protected.
> To the best of my knowledge the idea presented in the paper is novel. In my opinion, this work will influence future experiments, and can help solve the problem of a small topological gap in the planar Josephson junction. Moreover, the paper is well written. Assuming the authors can address the questions and comments listed below, I therefore recommend the publication of the paper in Physical Review Letters.

We thank the referee for the positive assessment of our work.

> My biggest concern has to do with the direction of the magnetic field as it is given in Eq. (1). The magnetic field considered in Eq. (1) does not penetrate the superconductor which is certainly reasonable. I think, however, that this is not consistent with the uniformly-directed magnetic field in the normal region (see B\sigma_x in Eq. (1a)). I would expect that in the zigzag geometry the magnetic field lines would be forced to curve in a zigzag manner to avoid approaching the superconductors in a direct angle. In this case, the direction of the Zeeman term in Eq. (1a) would depend on space. Can the authors give a compelling argument for why this would not affect the main results (or alternatively simulate a spatially-dependent magnetic field)?

We have performed a systematic investigation of the influence of the magnetic field direction on the device properties.
Because all the quasiparticle states are strongly coupled to the superconductor in the zigzag geometry, we observe that the magnetic field does not need to be aligned with the junction direction nearly as precisely as in a straight junction.
Specifically, we observe that even a 10-degree misalignment yields an induced gap of an order of magnitude larger than the gap in a straight junction with a perfectly aligned field.
We have also verified that a position-dependent magnetic field does not degrade the device performance.
We have reported these results these findings in the main text and added the detailed information in the appendix.
We thank the referee for the suggestion to investigate this phenomenon.

> At the end of section III, the authors explain that besides the effect of increasing the gap through eliminating modes traveling at grazing angles, the zigzag geometry has two more effects that causes the Majorana size to decrease: (1) reducing the Fermi velocity, and (2) increasing transmission to the SC. With regards to (1), I would comment that the Fermi velocity that enters the formula \xi=v_F/E_gap is the electron velocity before introducing superconductivity. The fact that the bandstructure becomes flatter is not a testament to the Fermi velocity becoming smaller since superconductivity is already present. With regards to (2), it seems to me that this is exactly the original effect that increases the gap, and should not be counted as a separate effect. Indeed, the transmission of electrons traveling at grazing angles is originally very small (it’s not zero though due to the uncertainty principle), and it becomes bigger since the incident angle is now more direct.

After the Eq. (2), we have added a remark that this equation follows from an avoided crossing shape of the dispersion relation near the Fermi momentum.
This relation does not assume anything about the nature of v_F or E_gap.

In order to distinguish the effect of the increased interface transparency from the effect of removing long trajectories, we have performed an additional simulation of a device with the same angle of the zigzag boundaries, but a much smaller amplitude of the modulation.
While we do observe a gap enhancement in this device, consistent with the mechanism (2) mentioned by the referee, this enhancement is much smaller than in the device with a large modulation.
Therefore, we conclude that the two mechanisms are independent, however, the removal of the long trajectories is the dominant mechanism for the gap enhancement.
We have added the details of the additional simulation to the supplementary information.

> Some smaller issues:
> Can the authors clarify the following statement from the introduction: “However, low filling requires precise knowledge of the system and is more sensitive to disorder or microscopic inhomogeneities”.

We thank the referee for pointing us to this vague sentence.
We have changed it to "However, tuning the system to a low chemical potential requires precise knowledge of the band positions and makes the device more sensitive to disorder or microscopic inhomogeneities."

> In the Discussion section, the authors mention that interface disorder can still cause a soft gap. Wasn’t that simulated in Fig. (3d)? A similar statement is made about a multimode junction. Does this refer to transverse modes in the z direction (perpendicular to the 2DEG)? because the system considered in the paper is indeed a multimode junction (due to the finite size in the y direction - W). Perhaps I didn’t understand the idea of the paragraph.

The reference 28 considers a variation in the coupling strength between a semiconductor and a superconductor that has a correlation length longer than the induced coherence length.
This scenario is therefore, very different from the short length scale roughness that we considered in Fig. 3(d).
Reference 30 describes a multimode mechanism and is about how the conductance looks rather than about the actual gap.


## List of changes
Apart from some minor grammatical and spelling errors, the amendments we made are:
* Rephrased section II.
* In section IV, we reformulated the following sentences:
  * >~~To reduce the finite size effects in determining the Majorana size $\xi_\textrm{M}$ in a zigzag system, we introduce a particle-hole symmetry breaking potential $V \sigma_0 \tau_0$ on one edge, such that one of the Majorana states is pushed away from zero energy.~~ When determining the Majorana size $\xi_\textrm{M}$ in a zigzag system, we reduce the finite size effects by introducing a particle-hole symmetry breaking potential $V \sigma_0 \tau_0$ on one edge, such that one of the Majorana states is pushed away from zero energy.
  * >~~The small topological gap combined with the high velocity result in a large Majorana size~~ This is a result of the small topological gap combined with the quasiparticle velocity $v \approx v_\textrm{F}$ yielding a large Majorana size
  * >~~The wave function extends to the center of the system, resulting in highly overlapping Majoranas and a Majorana coupling $E_\textrm{M}$ comparable to $E_\textrm{gap}$.~~ This result follows from an avoided crossing shape of the dispersion relation near the Fermi momentum. Therefore, in straight junctions the wave function extends to the center of the system, resulting in highly overlapping Majoranas and a Majorana coupling $E_\textrm{M}$ comparable to $E_\textrm{gap}$.
* In section V we refer to Appendix B where we compare the effects of magnetic field misalignment between a zigzag and a straight geometry.
* In section V we removed the sentence "We also observe additional gap closings due to the BDI symmetry."
* In section VI we moved the paragraph regarding experimental verification to the end.
* We added the following Appendices:
  * Appendix A, which details the implementation of a zigzag geometry in a device with a single superconductor.
  * Appendix B, which outlines the robustness of the topological gap in a zigzag device under misaligned magnetic field.
  * Appendix C, which demonstrates that the increased transparency of the NS interface alone does not explain the order of magnitude increase in topological gap.
