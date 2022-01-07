# pytorch workflow

1. 데이터셋을 불러온다
2. 데이터로더를 형성해 배치 사이즈를 입력
3. 학습에 사용할 CPU나 GPU 장치를 얻습니다.
4. 모델을 정의합니다
5. 로스펑션과 옵티마이저를 생성합니다
6. 학습과 테스트를 정의합니다
7. 학습과 테스트
8. 모델을 저장합니다.


# 파이토치 RNN 레퍼런스
https://data-science-hi.tistory.com/190

$$
\ket{\Phi^+}_{AB}\ket{\Phi^+}_{CD} 
= \frac{1}{2}(\ket{0}_{A}\ket{00}_{BC}\ket{0}_{D} 
+ \ket{0}_{A}\ket{01}_{BC}\ket{1}_{D} + \ket{1}_{A}\ket{10}_{BC}\ket{0}_{D} + \ket{1}_{A}\ket{11}_{BC}\ket{1}_{D})
$$

Bell State Measurement for $\ket{\Phi^+}_{BC}$
$$
I\otimes \bra{\Phi^+}_{BC}\otimes I
$$

$$
I\otimes \bra{\Phi^+}_{BC}\otimes I\ket{\Phi^+}_{AB}\ket{\Phi^+}_{CD}
= \frac{1}{2\sqrt{2}}[
    \ket{0}(\braket{00|00}+\braket{11|00})\ket{0}
    +\ket{0}(\braket{00|01}+\braket{11|01})\ket{1}
    +\ket{1}(\braket{00|10}+\braket{11|10})\ket{0}
    +\ket{1}(\braket{00|11}+\braket{11|11})\ket{1}
 ] 
$$
$$
= \frac{1}{2\sqrt{2}}[\ket{00}_{AD}+\ket{11}_{AD}] = \frac{1}{2}\ket{\Phi^+}_{AD}
$$