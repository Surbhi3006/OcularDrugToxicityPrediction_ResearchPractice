def findDecision(obj): #obj[0]: C Log P, obj[1]: TPSA, obj[2]: Molecular Weight, obj[3]: nON, obj[4]: nOHNH, obj[5]: ROTB, obj[6]: Molecular Volume
   # {"feature": "nON", "instances": 136, "metric_value": 0.9923, "depth": 1}
   if obj[3]>2.0:
      # {"feature": "Molecular Volume", "instances": 128, "metric_value": 0.9786, "depth": 2}
      if obj[6]<=533.8724827295733:
         # {"feature": "nOHNH", "instances": 122, "metric_value": 0.9617, "depth": 3}
         if obj[4]>0.0:
            # {"feature": "Molecular Weight", "instances": 119, "metric_value": 0.9505, "depth": 4}
            if obj[2]<=460.44347707484883:
               # {"feature": "C Log P", "instances": 97, "metric_value": 0.8386, "depth": 5}
               if obj[0]<=2.589759219756732:
                  # {"feature": "TPSA", "instances": 86, "metric_value": 0.6931, "depth": 6}
                  if obj[1]>36.42:
                     # {"feature": "ROTB", "instances": 85, "metric_value": 0.6723, "depth": 7}
                     if obj[5]>3.0:
                        return '1.0'
                     elif obj[5]<=3.0:
                        return '1.0'
                     else:
                        return '1.0'
                  elif obj[1]<=36.42:
                     return '0.0'
                  else:
                     return '0.0'
               elif obj[0]>2.589759219756732:
                  # {"feature": "TPSA", "instances": 11, "metric_value": 0.4395, "depth": 6}
                  if obj[1]>31.64:
                     return '0.0'
                  elif obj[1]<=31.64:
                     # {"feature": "ROTB", "instances": 2, "metric_value": 1.0, "depth": 7}
                     if obj[5]>6.0:
                        return '0.0'
                     elif obj[5]<=6.0:
                        return '1.0'
                     else:
                        return '1.0'
                  else:
                     return '1.0'
               else:
                  return '0.0'
            elif obj[2]>460.44347707484883:
               # {"feature": "C Log P", "instances": 22, "metric_value": 0.684, "depth": 5}
               if obj[0]>-5.33:
                  # {"feature": "TPSA", "instances": 21, "metric_value": 0.5917, "depth": 6}
                  if obj[1]<=139.7852380952381:
                     # {"feature": "ROTB", "instances": 13, "metric_value": 0.7793, "depth": 7}
                     if obj[5]>5.0:
                        return '0.0'
                     elif obj[5]<=5.0:
                        return '0.0'
                     else:
                        return '0.0'
                  elif obj[1]>139.7852380952381:
                     return '0.0'
                  else:
                     return '0.0'
               elif obj[0]<=-5.33:
                  return '1.0'
               else:
                  return '1.0'
            else:
               return '0.0'
         elif obj[4]<=0.0:
            return '0.0'
         else:
            return '0.0'
      elif obj[6]>533.8724827295733:
         return '0.0'
      else:
         return '0.0'
   elif obj[3]<=2.0:
      return '0.0'
   else:
      return '0.0'
