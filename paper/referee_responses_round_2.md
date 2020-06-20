---
urlcolor: blue
geometry:
  margin=1in
header-includes:
- \usepackage{tcolorbox}
- \newtcolorbox{myquote}{colback=gray!5!white, colframe=gray!75!black}
- \renewenvironment{quote}{\begin{myquote}}{\end{myquote}}
---

Dear Dr. Samindranath Mitra,

We would like to resubmit our revised manuscript where we address the referee C's remarks, for which we have done a set of new simulations.
We have added one of these simulations' result as a paragraph to the Supplemental Material: "Zigzag devices and disorder".
We also address the two questions below.

Sincerely,  
The authors

## Second Report of Referee B -- LD17005/Laeven

> I have read the report of Referee A and the reply to both reports by the authors. I'm satisfied with the authors' reply to my comments and with the changes made. I believe this work will have an impact on upcoming experiments in topological superconductivity in the near future. I therefore recommend the publication of the paper in PRL.

We thank the referee for the recommendation.

## Report of Referee C -- LD17005/Laeven

> I read with interest the manuscript entitled “Enhanced proximity effect in zigzag-shaped Majorana Josephson junctions” by Tom Laeven et al. together with the others referees reports. The Authors propose new geometry of Josephson – zigzag or sine-like shaped - which enhances size of induced topological gap by more than order of magnitude comparing to the standard straight geometry. This results in shorter coherence length of the “Majoranas” and in turns leads to increase of robustness against disorder which is of great interest for physical implementation. In my opinion Authors managed to fully answer the other referees questions.
>
> In order to judge if paper is appropriate for PRL let’s analyze the results in the light of PRL's acceptance criteria. The work should:
>
> 1) Open a new research area, or a new avenue within an established area.
>
> The work does not open new research area
>
> 2) Solve, or make essential steps towards solving, a critical problem.
>
> I think that proposing a new geometry (zigzag, sine type) of Josephson junction which enhances the induced topological gap of the system by more than order of magnitude can be considered as “making essential steps towards solving, a critical problem.” It is worth to mention that proposed setup has been already realized experimentally http://meetings.aps.org/Meeting/MAR20/Session/L60.6.
>
> 3) Introduce new techniques or methodologies with significant impact.
>
> The work does not really present new techniques or methodologies...
>
> 4) Be of unusual intrinsic interest to PRL's broad audience.
>
> The work touches rather specific and greatly explored field of Majorana based systems.
> Overall I think the manuscript matches criteria 2) and this should be sufficient for publication in PRL, thus I can recommend the paper for publication if the Authors could answer and clarify the following matter:
>
> i) Is it important that in proposed setup (Figure 3) the zigzag/sine-like geometry pattern of normal region has a reflection symmetry with respect to $x=x_M$ axis, where $x_M$ denotes middle of the system in the $x$ direction? I wonder if taking zigzag pattern, for which middle of the junction in $y$ direction is different at two ends of the system (e.g. by taking scenario with zigzag endings like on Fig. 1), would affect the size of induced gap and presence of Majoranas significantly?
>
> ii) I wonder if combing the effects of zigzag geometry and weak disorder presented in ref [22] (Arbel Haim and Ady Stern Phys. Rev. Lett. 122, (2019)) could further increase the size of induced gap?

We would like to thank referee C for his/her evaluation. To answer the questions, we have performed two numerical simulations.

Specifically, to determine whether the ends of the zigzag region matter we have created systems that end before a full zigzag period (see the attached figure). We observe that this does not influence any of our conclusions. We believe this was expected as we make the argument that the proposed mechanism for the gap increase (cutting off of long trajectories) is not influenced by its endings.

To answer the question regarding the combination of disorder and zigzag, we have simulated a straight and zigzag system for an increasing mean free path and calculated the average energy gap (averaged over disorder realizations). For the parameters we chose, at $\phi=0$ we observe that an optimal value of the mean free path results in an enhancement that is at best equal to that of the zigzag device. At $\phi=\pi$, we observe that zigzag geometry outperforms disorder. We have added a paragraph to the appendix of the manuscript that describes our findings.

## List of changes

The following changes were made:

* Added a paragraph to the Supplemental Material: "Zigzag devices and disorder".
* Refer to this paragraph in the main text.
