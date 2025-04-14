import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../auth/otp_service.dart';

class UserRepository {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final OTPService _otpService = OTPService();

  Future<void> sendPhoneOTP(String phoneNumber) async {
    await _otpService.sendPhoneOTP(phoneNumber);
  }

  Future<void> sendEmailOTP(String email) async {
    await _otpService.sendEmailOTP(email);
  }

  Future<void> verifyPhoneOTP(String phoneNumber, String otp) async {
    await _otpService.verifyPhoneOTP(phoneNumber, otp);
  }
  Future<void> login(String email, String password) async {
    // Implement your login logic here
    // Example: Perform a Firebase login or other authentication logic.

    try {
      // Assuming you are using Firebase Authentication
      await FirebaseAuth.instance.signInWithEmailAndPassword(
        email: email,
        password: password,
      );
      print('User logged in successfully');
    } catch (e) {
      print('Login failed: $e');
      throw e; // Handle error as per your requirement
    }
  }
  Future<bool> verifyEmailOTP(String otp) async {
    return await _otpService.verifyEmailOTP(otp);
  }

  Future<void> registerUser(String email, String password, String phoneNumber) async {
    await _firestore.collection('users').add({
      'email': email,
      'password': password,  // For security, encrypt this in production
      'phoneNumber': phoneNumber,
    });
  }
}
