output "ec2_public_ip" {
  description = "The public IP of the web server"
  value       = aws_instance.app_server.public_ip
}

output "security_group_id" {
  value = aws_security_group.detector_sg.id
}