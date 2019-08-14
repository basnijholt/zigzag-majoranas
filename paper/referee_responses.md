> # Report of Referee A -- LD17005/Laeven
> The manuscript discusses numerical calculations related to how to improve the energy gap in 2D planar Josephson junctions.
> The manuscript considers 2D SNS junctions within clean-limit BdG tight-binding model, used also in previous studies of the same system. Specific N region geometry is considered, and found to improve the energy gap. The argument why this occurs appears clear, and the results valid. Improving energy gaps likely is useful, but it's less clear this is the main/only problem faced in the experiments.
> The effect of long trajectories in ballistic systems is itself not surprising or new, but optimizing the geometry for this in the specific context of Majorana realizations with planar SNS was apparently not considered before. The closer study of this is the main new contribution here.
> However, the manuscript appears better suited for publication in PRB. Although optimizing parameters in a previously discussed model is interesting and probably useful for future experiments of this type, it does not appear a significant advance of wide interest to a broad audience.

This is a conceptually new idea, overlooked in a very mature field, which makes it exciting and it seems to resonate with a lot of people.
Even though the effect of long trajectories was known to DeGennes, the idea of optimizing the geometry to eliminate long trajectories has not previously been considered even outside of the Majorana context.


> # Report of Referee B -- LD17005/Laeven
> The authors consider a planar Josephson junction in a zigzag geometry and propose it as an improved realization of a one-dimensional topological superconductor. As they show using numerical simulations, the zigzag geometry increases the induced superconducting gap and thereby decreases the spatial size of the Majorana wave functions, making them more protected.
> To the best of my knowledge the idea presented in the paper is novel. In my opinion, this work will influence future experiments, and can help solve the problem of a small topological gap in the planar Josephson junction. Moreover, the paper is well written. Assuming the authors can address the questions and comments listed below, I therefore recommend the publication of the paper in Physical Review Letters.
> My biggest concern has to do with the direction of the magnetic field as it is given in Eq. (1). The magnetic field considered in Eq. (1) does not penetrate the superconductor which is certainly reasonable. I think, however, that this is not consistent with the uniformly-directed magnetic field in the normal region (see B\sigma_x in Eq. (1a)). I would expect that in the zigzag geometry the magnetic field lines would be forced to curve in a zigzag manner to avoid approaching the superconductors in a direct angle. In this case, the direction of the Zeeman term in Eq. (1a) would depend on space. Can the authors give a compelling argument for why this would not affect the main results (or alternatively simulate a spatially-dependent magnetic field)?

The flatness of the superconductors may diminish, but in order to consider inhomogeneity, we exaggerate the effect by considering a magnetic field which is always perfectly aligned with the nanowire, corresponding to perfect expulsion.
We have added a section to the appendix and done several simulations to answer these concerns.
Specifically, we have done a simulation of a rotating magnetic field and observe that the zigzag device is much more resilient against misaligned magnetic fields than a straight device.
Additionally, we have done a simulation where the magnetic field follows the curvature of the device as the referee suggested.
Both these simulations aim to show that the concerns raised by the referee are not a problem.

> At the end of section III, the authors explain that besides the effect of increasing the gap through eliminating modes traveling at grazing angles, the zigzag geometry has two more effects that causes the Majorana size to decrease: (1) reducing the Fermi velocity, and (2) increasing transmission to the SC. With regards to (1), I would comment that the Fermi velocity that enters the formula \xi=v_F/E_gap is the electron velocity before introducing superconductivity. The fact that the bandstructure becomes flatter is not a testament to the Fermi velocity becoming smaller since superconductivity is already present. With regards to (2), it seems to me that this is exactly the original effect that increases the gap, and should not be counted as a separate effect. Indeed, the transmission of electrons traveling at grazing angles is originally very small (it’s not zero though due to the uncertainty principle), and it becomes bigger since the incident angle is now more direct.

After the Eq. (2) we have added a remark that this equation follows from an avoided crossing shape of the dispersion relation near the Fermi momentum. This relation does not assume anything about the nature of v_F or of E_gap.

The separate two phenomena, we have simulated a straight junction where the boundary is also corrugated so that the incident angle of the grazing trajectories is higher.
In the last paragraph of the appendix we note that the gap is indeed increased (2x) due to that phenomenon, although the order of magnitude change only occurs when the long trajectories are cut of by the geometry.

> Some smaller issues:
> Can the authors clarify the following statement from the introduction: “However, low filling requires precise knowledge of the system and is more sensitive to disorder or microscopic inhomogeneities”.

We thank the referee for pointing us to this vague sentence. We have changed it to "However, tuning the system to a low chemical potential requires precise knowledge of the band positions and makes the device more sensitive to disorder or microscopic inhomogeneities."

> In the Discussion section, the authors mention that interface disorder can still cause a soft gap. Wasn’t that simulated in Fig. (3d)? A similar statement is made about a multimode junction. Does this refer to transverse modes in the z direction (perpendicular to the 2DEG)? because the system considered in the paper is indeed a multimode junction (due to the finite size in the y direction - W). Perhaps I didn’t understand the idea of the paragraph.

The reference 29 considers a change in the barrier transparency and a longer correlation length and is therefore not the same. Reference 31 describes a multimode mechanism and is about how the conductance looks rather than about the actual gap.
