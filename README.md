# BLE-POLA

BLE POLA, or Bluetooth Low Energy Pattern of Life Analysis, is a complex topic that my team decided to hopefully understand. Our initial question was, "How can we monitor and protect BLE devices and connections in an age of abundant Cybersecurity threats?" 

Our first answer was to monitor and research BLE traffic in order to identify malicious behavior. But how do we effictively do this?

Well, in our project my team and I conducted various BLE attacks on wireless devices to collect malicious packets, utilizing Flipper Zeros. Our goal was to analyze this data using Machine Learning (ML) to classify different types of BLE packets as either malicious or benign.

## What is BLE?

As we started our initial research into this project, we of course wanted to begin with the basics. The basics being looking into what BLE is and how it works. The Bluetooth Technology website describes BLE as a ”radio designed for very low power operation.” More specifically, BLE is a protocol that allows short-range communication between low-power IoT devices such as smartphones, watches, and headphones. BLE works by using 40, two Megahertz channels to transmit data over radio frequency from one device to another. Despite its effectiveness in transmitting data while minimizing power use Bluetooth 4.0, otherwise known as Bluetooth Low Energy, remains vulnerable to a range of Cybersecurity attacks.

## BLE Attack Vectors

Following our general research into BLE, we started to look into specific Cybersecurity attack vectors that are primarily used on these types of connections. Some of the most common and interesting attacks being Bluejacking, Bluesnarfing, and Braktooth. For our project we decided to experiment with Bluejacking and Bad KB as they were the easiest to get working effectively and legally on our Flipper Zeros.

## Machine Learning

Continuing on with the next project section, our research directed us toward the use of Machine Learning and how to find the best algorithm to use. ML is a vast subject which means selecting the right model is crucial to our projects success. Many models don't utilize the correct algorithms for our use case, therefore we would have to find a model that is either multi faceted or is specifically used for classification. Our research was narrowed down to the following algorithms which will be trained and tested in this github repository: Logistic Regression, Decision Trees, Random Forest, and Support Vector Machine (SVM).

# Whats Needed?

1. A computer

# Setting Up The Environment...

1. Navigate into the environment folder with "cd sklearn-env"
2. Install all required dependecies with "pipenv install"
3. Start the environment with "pipenv shell"
4. To close the environement run "exit"
