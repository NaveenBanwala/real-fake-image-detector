provider "aws" {
  region = var.aws_region
}

resource "aws_security_group" "detector_sg" {
  name        = "real-fake-detector-sg-v3"
  description = "Allow inbound traffic for App and SSH"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "app_server" {
  ami           = "ami-0522ab6e1ddcc7055" # Mumbai Ubuntu 22.04
  instance_type = "t3.micro"            # Staying in Free Tier
  key_name      = "NaveenBanwala"
  vpc_security_group_ids = [aws_security_group.detector_sg.id]
  user_data_replace_on_change = true

  # Install K3s with "No Extras" to save RAM
 # Install K3s with "No Extras" and allow Port 8000 for NodePort services
  user_data = <<-EOF
              #!/bin/bash
              # This script installs K3s and tells it to allow Port 8000
              curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server --disable traefik --disable metrics-server --kube-apiserver-arg=service-node-port-range=8000-32767" sh -
              sleep 30
              sudo chmod 644 /etc/rancher/k3s/k3s.yaml
              EOF

  tags = { Name = "Real-Fake-K8s-Final-Deployment" }
}

output "instance_ip" {
  value = aws_instance.app_server.public_ip
}