import 'package:flutter/material.dart';

const primaryColor = Color(0xFF6C63FF);
const secondaryColor = Color(0xFFF1F1F1);
const accentColor = Color(0xFFFF6584);

const inputDecoration = InputDecoration(
  filled: true,
  fillColor: secondaryColor,
  enabledBorder: OutlineInputBorder(
    borderSide: BorderSide(color: Colors.transparent),
    borderRadius: BorderRadius.all(Radius.circular(12.0)),
  ),
  focusedBorder: OutlineInputBorder(
    borderSide: BorderSide(color: primaryColor),
    borderRadius: BorderRadius.all(Radius.circular(12.0)),
  ),
);

 var buttonStyle = ButtonStyle(
  backgroundColor: MaterialStateProperty.all(primaryColor),
  shape: MaterialStateProperty.all(
    RoundedRectangleBorder(borderRadius: BorderRadius.all(Radius.circular(10))),
  ),
  padding: MaterialStateProperty.all(EdgeInsets.symmetric(vertical: 16)),
);
