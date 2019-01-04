import matplotlib as mpl
import numpy as np
mpl.use('agg')
import matplotlib.pyplot as plt

x = ['Token', 'Type', 'GloVe', 'ELMO', 'Hand', 'All']
ind = np.arange(len(x))
y_part = [39.2, 51.6, 44.8, 58.2, 54.8, 58.0]
y_kind = [26.0, 42.0, 33.1, 47.9, 45.6, 48.0]
y_abs = [49.2, 34.4, 46.9, 55.8, 51.6, 56.2]

plt.figure()
plt.suptitle('Argument correlation scores')
ax = plt.subplot(111)
# ax.set_xlabel('Model features')
ax.set_ylabel('Pearson correlation')
ax.bar(ind - 0.2, y_part, width=0.2, color='b', align='center', label='Particular')
ax.bar(x, y_kind, width=0.2, color='g', align='center', label='Kind')
ax.bar(ind + 0.2, y_abs, width=0.2, color='r', align='center', label='Abstract')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('bars-arg-pear.png')


x = ['Token', 'Type', 'GloVe', 'ELMO', 'Hand', 'All']
ind = np.arange(len(x))
y_part = [16.3, 20.9, 20.3, 27.5, 24.7, 28.5]
y_hyp = [13.8, 38.3, 22.9, 42.2, 38.8, 42.0]
y_dyn = [33.2, 31.5, 29.4, 38.3, 37.5, 38.8]

plt.figure()
plt.suptitle('Predicate correlation scores')
ax = plt.subplot(111)
# ax.set_xlabel('Model features')
ax.set_ylabel('Pearson correlation')
ax.bar(ind - 0.2, y_part, width=0.2, color='b', align='center', label='Particular')
ax.bar(x, y_hyp, width=0.2, color='g', align='center', label='Hypothetical')
ax.bar(ind + 0.2, y_dyn, width=0.2, color='r', align='center', label='Dynamic')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('bars-pred-pear.png')

x = ['Token', 'Type', 'GloVe', 'ELMO', 'Hand', 'All']
ind = np.arange(len(x))
y_part = [7.9, 13.1, 9.7, 15.7, 13.5, 15.8]
y_kind = [2.1, 8.9, 2.6, 11.6, 10.8, 11.4]
y_abs = [14.1, 6.3, 11.9, 17.3, 15.2, 17.7]

plt.figure()
plt.suptitle('Argument R1 scores')
ax = plt.subplot(111)
ax.set_xlabel('Model features')
ax.set_ylabel('R1')
ax.bar(ind - 0.2, y_part, width=0.2, color='b', align='center', label='Particular')
ax.bar(x, y_kind, width=0.2, color='g', align='center', label='Kind')
ax.bar(ind + 0.2, y_abs, width=0.2, color='r', align='center', label='Abstract')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('bars-arg-r1.png')


x = ['Token', 'Type', 'GloVe', 'ELMO', 'Hand', 'All']
ind = np.arange(len(x))
y_part = [1.2, 2.7, 2.2, 3.9, 3.4, 4.5]
y_hyp = [0, 3.7, 0, 6.0, 2.9, 7.2]
y_dyn = [6.0, 5.4, 4.6, 7.8, 7.9, 8.4]

plt.figure()
plt.suptitle('Predicate R1 scores')
ax = plt.subplot(111)
# ax.set_xlabel('Model features')
ax.set_ylabel('R1')
ax.bar(ind - 0.2, y_part, width=0.2, color='b', align='center', label='Particular')
ax.bar(x, y_hyp, width=0.2, color='g', align='center', label='Hypothetical')
ax.bar(ind + 0.2, y_dyn, width=0.2, color='r', align='center', label='Dynamic')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('bars-pred-r1.png')
