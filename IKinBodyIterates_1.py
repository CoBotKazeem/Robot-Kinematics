import numpy as np
import modern_robotics as mr
import csv

def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev):
  """Computes inverse kinematics in the body frame for an open chain robot,
     including full iteration status output to console and joint angles to csv

  :param Blist: The joint screw axes in the end-effector frame when the
                manipulator is at the home position, in the format of a
                matrix with axes as the columns
  :param M: The home configuration of the end-effector
  :param T: The desired end-effector configuration Tsd
  :param thetalist0: An initial guess of joint angles that are close to
                     satisfying Tsd
  :param eomg: A small positive tolerance on the end-effector orientation
               error. The returned joint angles must give an end-effector
               orientation error less than eomg
  :param ev: A small positive tolerance on the end-effector linear position
             error. The returned joint angles must give an end-effector
             position error less than ev
  :return thetalist: Joint angles that achieve T within the specified
                     tolerances,
  :return success: A logical value where TRUE means that the function found
                   a solution and FALSE means that it ran through the set
                   number of maximum iterations without finding a solution
                   within the tolerances eomg and ev.
  Uses an iterative Newton-Raphson root-finding method.
  The maximum number of iterations before the algorithm is terminated has
  been hardcoded in as a variable called maxiterations. It is set to 20 at
  the start of the function, but can be changed if needed.

  Example Input:
      Blist = np.array([[0, 0, -1, 2, 0,   0],
                        [0, 0,  0, 0, 1,   0],
                        [0, 0,  1, 0, 0, 0.1]]).T
      M = np.array([[-1, 0,  0, 0],
                    [ 0, 1,  0, 6],
                    [ 0, 0, -1, 2],
                    [ 0, 0,  0, 1]])
      T = np.array([[0, 1,  0,     -5],
                    [1, 0,  0,      4],
                    [0, 0, -1, 1.6858],
                    [0, 0,  0,      1]])
      thetalist0 = np.array([1.5, 2.5, 3])
      eomg = 0.01
      ev = 0.001
  Output:
      Iteration 0 :
      joint vector : [1.5 2.5 3. ]
      SE(3) end−effector config : 
      [[-0.071  0.997  0.    -4.489]
       [ 0.997  0.071  0.     4.318]
       [ 0.     0.    -1.     1.7  ]
       [ 0.     0.     0.     1.   ]]
      error twist V_b : [ 0.          0.          0.07079633 -0.30008633 -0.52232685  0.0142    ]
      angular error magnitude ∣∣omega_b∣∣ : 0.0707963267948966
      linear error magnitude ∣∣v_b∣∣ : 0.6025601901965655

      Iteration 1
      joint vector : [1.58239319 2.97475147 3.15307873]
      SE(3) end−effector config : 
      [[-0.     1.     0.    -4.974]
       [ 1.     0.     0.     3.942]
       [ 0.     0.    -1.     1.685]
       [ 0.     0.     0.     1.   ]]
      error twist V_b : [ 0.          0.          0.00011079  0.05769165 -0.02557985 -0.00110787]
      angular error magnitude ∣∣omega_b∣∣ : 0.00011078732257497602
      linear error magnitude ∣∣v_b∣∣ : 0.06311800417847872

      Iteration 2
      joint vector : [1.57073819 2.999667   3.14153913]
      SE(3) end−effector config : 
      [[ 0.     1.     0.    -5.   ]
       [ 1.    -0.     0.     4.   ]
       [ 0.     0.    -1.     1.686]
       [ 0.     0.     0.     1.   ]]
      error twist V_b : [ 0.00000000e+00  0.00000000e+00 -4.60870782e-06 -2.90646809e-04
       -3.33010395e-04  4.60870782e-05]
      angular error magnitude ∣∣omega_b∣∣ : 4.608707824018659e-06
      linear error magnitude ∣∣v_b∣∣ : 0.0004444046688059929
  """

  thetalist = np.array(thetalist0).copy()
  i = 0
  maxiterations = 20
  
  # Setup up CSV output
  filename = "iterates.csv"
  outfile = open(filename, 'w')
  csvwriter = csv.writer(outfile)
  csvwriter.writerow(thetalist)

  print("Iteration " + str(i) + " :")
  print("joint vector : " + str(thetalist))

  # Break out FK to allow dump of EE pose
  EEconfig = mr.FKinBody(M, Blist, thetalist)
  print("SE(3) end−effector config : \n" + str(np.round(EEconfig,3)))

  Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(EEconfig), T)))

  # Dump error twist
  print("error twist V_b : " + str(Vb))

  # Break out errors for console dump
  angErrorMag = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
  linErrorMag = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
  print("angular error magnitude ∣∣omega_b∣∣ : " + str(angErrorMag))
  print("linear error magnitude ∣∣v_b∣∣ : " + str(linErrorMag) + "\n")

  err = angErrorMag > eomg or linErrorMag > ev

  while err and i < maxiterations:

      print("Iteration " + str(i+1))      

      thetalist = thetalist \
                  + np.dot(np.linalg.pinv(mr.JacobianBody(Blist, thetalist)), Vb)
      # Write joint angles to CSV and console
      csvwriter.writerow(thetalist)
      print("joint vector : " + str(thetalist))  

      # Break out FK to allow dump of EE pose
      EEconfig = mr.FKinBody(M, Blist, thetalist)
      print("SE(3) end−effector config : \n" + str(np.round(EEconfig,3)) )
                                            
      i = i + 1
      Vb \
      = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(EEconfig), T)))
  
      # Dump error twist
      print("error twist V_b : " + str(Vb))

      # Break out errors for console dump
      angErrorMag = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
      linErrorMag = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
      print("angular error magnitude ∣∣omega_b∣∣ : " + str(angErrorMag))
      print("linear error magnitude ∣∣v_b∣∣ : " + str(linErrorMag) + "\n")

      err = angErrorMag > eomg or linErrorMag > ev

  # Don't forget to close csv file
  outfile.close()

  return (thetalist, not err)