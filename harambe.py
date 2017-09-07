from sklearn.tree import DecisionTreeClassifier

#raw data [height, weight, shoe size]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42],
     [181, 85, 43]]

Y = ['Boy', 'Boy', 'Girl', 'Girl', 'Boy', 'Boy', 'Girl', 'Girl', 
     'Girl', 'Boy', 'Boy']


#classifier function declaration
clf = DecisionTreeClassifier()

#classifier function fit to data
clf = clf.fit(X,Y)


#predictions for boy and girl using new parameters
predictionBoy = clf.predict([185, 85, 45])
predictionGirl = clf.predict([150, 59, 37])

#print result of prediction
print (predictionBoy)
print (predictionGirl)
