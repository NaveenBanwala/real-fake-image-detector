output "instance_ip" {
  description = "The public IP of the web server"
  value       = aws_instance.app_server.public_ip
}

output "security_group_id" {
  description = "The ID of the new app security group"
  value       = aws_security_group.app_sg.id
}