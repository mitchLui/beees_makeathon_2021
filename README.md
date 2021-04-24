# 1. BEEES Make-a-thon 2021

This repository stores the project created for the [University of Bristol BEEES Make-a-thon 2021][1].

- [1. BEEES Make-a-thon 2021](#1-beees-make-a-thon-2021)
  - [1.1. About the Hackathon](#11-about-the-hackathon)
  - [1.2. Our Team](#12-our-team)
  - [1.3. Our Project](#13-our-project)
  - [1.4. Tech Stack](#14-tech-stack)
  - [1.5. Deployment](#15-deployment)
  - [1.6. Usage](#16-usage)

---
## 1.1. About the Hackathon

The BEEES Make-a-thon is a 48-hour hackathon that is open to all courses and years of students in the University of Bristol. 

The theme for 2021 is **COVID-19**.

---

## 1.2. Our Team

Our team (named Rogue One after the fact that we were team 1 and Star Wars is awesome) consists of 6 people from the University of Bristol, all of whom are from Hong Kong:

- [Mitch Lui][2] (Computer Science, 1st year)
- [Otis Lee][3] (Electrial and Electronic Engineering, 1st year)
- [Javis Lo][4] (Aerospace Engineering, 1st year)
- [Ken Young][5] (Aerospace Engineering, 1st year)
- [Dominic Chu][6] (Finance, 1st year)

---

## 1.3. Our Project

We decided to create a mask detection application to ensure that people are wearing masks correctly when entering stores or indoor places where social distancing is not possible.

A Machine Learning model based on this [article][7] was adapted and improved upon by introducing a larger dataset, which allowed us to reach a high degree of accuracy when trying to detect if people were wearing masks correctly.

The model is then connected to a webcam to get a live feed and detect if a person is wearing a mask. If they are wearing a mask, a connected Arduino board will let the person know they can enter by disabling the LED which is always on when the people in front of the webcam are not wearing masks correctly.

---

## 1.4. Tech Stack



---

## 1.5. Deployment



---

## 1.6. Usage


---

[1]: https://www.beees.co.uk/make-a-thon-2021-announcement/
[2]: https://www.linkedin.com/in/mitchlui/
[3]: https://www.linkedin.com/in/otis-lee-9154a91ba/
[4]: https://www.linkedin.com/in/yat-chung-javis-lo-807611200/
[5]: https://www.linkedin.com/in/ken-y-6b6379142/
[6]: https://www.linkedin.com/in/dominic-chu-544966178/
[7]: https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/