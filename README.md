# UE5-GetupControl
Get up motion generator without data(RL using SAC)

## 주제
SAC 강화학습 알고리즘, Strong to Weak 방법으로 기립 애니메이션 생성

## 참고 논문
Tao, T., Wilson, M., Gou, R., & van de Panne, M. (2022, August 7). Learning to Get Up. Special Interest Group on Computer Graphics and Interactive Techniques Conference Proceedings. ACM.

## 동기
현재 게임 속 캐릭터의 기립 모션이 부자연스러운 경우가 많다. 누워있는 모습의 형태는 다양하지만, 모든 초기 상태에 대해서 애니메이션을 제작할 수 없기 때문이다. 강화학습으로 어떤 초기 상태에서라도 물리적으로 알맞게 일어날 수 있는 액터를 학습할 수 있다면 다양한 초기 상태에 대해서 애니메이션을 제작할 수 있을 것이다.

## 내용
참고 논문에서는 mujoco 환경에서 학습을 하고 시각화했다.
우리는 게임 제작에 많이 쓰이는 언리얼 엔진에서 UE5 마네킹으로 학습시킨다.

## 결과
어떻게 누워있든지 상관없이 일어날 수 있는 결과물이 나왔다. 결과물이 후처리가 필요한 퀄리티이긴 하지만, 더 긴 시간 학습시키거나 더 적절한 파라미터 값과 리워드를 사용한다면 더 좋은 결과물이 나올 것이다. 또한 참고한 논문에서 소개한 slow까지 적용한다면 자연스러운 모션이 나올 것이다.

## 기술의 장점
1. 학습 훈련 데이터 (기존 애니메이션, 영상)이 필요 없다.
2. 모션 캡쳐로는 쉽지 않은 다리 여러 개, 팔 여러 개, 긴 팔 등 인간이 아닌 비현실 개체에 대해서도 애니메이션 생성이 가능하다.
3. 기립 애니메이션 외에도 다양한 동작을 Reward 값의 변경만으로 학습시킬 수 있다. SAC와 Strong to Weak를 사용하여 학습 데이터 없이 빠르게 학습시킬 수 있었다.
##
### [소개영상](https://youtu.be/LTb6Gi-Ucxc)
##
### [언리얼프로젝트 다운받기](https://drive.google.com/file/d/1GaunUPkVFNOqFSoIIizr1BrxmcxNxFBB/view?usp=sharing)
##
### 프로젝트 버전
Unreal Engine 5.1.1
Python 3.8.16
## 사용법
1. 언리얼 프로젝트를 열고 test맵을 연다. 
2. 60프레임을 넘기지 못하게 하도록 60프레임 고정을 해줘야한다.(60프레임이 넘어가면 관절에 주는 힘들인 액션이 적용되는 속도가 달라져 일어나지 못한다.)
3. 이 저장소에 올려져있는 폴더를 파이썬으로 연다.(가상환경으로 여는 걸 추천한다. 패키지 버전과 파이썬 버전을 맞춰줘야되기 때문)
4. 파이썬 main.py을 실행한다. (그 전에 필요한 패키지들을 미리 설치해야한다. requirement.txt에 있는 것들을 설치해야한다.)
5. 파이썬이 제대로 실행하면 언리얼도 실행한다. 언리얼에서 TCP 연결에 실패했다고 하면 언리얼만 재실행한다.
6. 일어서려고 노력하는 언리얼 마네킹을 볼 수 있다.
## 활용법
1. 언리얼 프로젝트에서 초기 상태를 바꿔서 초기 state를 구하고 파이썬 main.py코드에서 state = [[ 로 되어있는 모든 state를 바꿔준다.
2. 약 80만번쯤하다보면 일어서기 시작한다. 만약 python log에서 Loss값들이 너무 커지는 경우 학습이 이상해진 경우이므로 재학습해야한다.(프레임이 끊기는 경우 학습이 망가진다. 언리얼 창을 띄우고 아무것도 하지 말아야 한다.)
3. 이상해지기전 가장 좋은 모델인 best model은 저장되어있으므로 학습했던 폴더를 load_dir에 경로를 알맞게 설정하여 이어서 학습시킨다. (load_dir엔 우리가 학습시켜놓은 standupfinal이 연결되어있다.)
4. 다른 모션들을 생성하고 싶다면 논문을 참고하여 언리얼에 구현된 Reward를 입맛에 맞게 변경하여 학습시켜본다.
