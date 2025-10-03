package com.dsa.tree;

class TreeRoot{

    public TreeRoot left,right ;
    public int data;

    public TreeRoot(int data,TreeRoot left,TreeRoot right){
        this.data = data;
        this.left = left;
        this.right = right;
    }

}

public class BinaryTree {

    public static TreeRoot root; // this become the root of the tree (Parent)

    public void add(int data){
        TreeRoot newRoot = new TreeRoot(data,null,null);

        // check if Tree is empty than root become null
        if(root == null){
            root = newRoot;
        }else{

            // get parent
            TreeRoot current = root; TreeRoot parent = root;
            while(current != null){
                parent = current;
                if(newRoot.data > current.data){
                    current = current.right;
                }else {
                    current = current.left;
                }
            }

            // found parent now check where to add
            if(newRoot.data > parent.data){
                parent.right = newRoot;
            }else {
                parent.left = newRoot;
            }
        }

    }

    // Traverse Tree Inorder
    public void inorder(TreeRoot root){
        TreeRoot r = root;
            if (root.left != null) {
                r = root.left;
                inorder(r);
            }
            System.out.println(" data in tree " + root.data);
            if (root.right != null) {
                r = root.right;
                inorder(r);
            }
    }

    // Delete node from tree

    public TreeRoot delete(TreeRoot root, int key){

        if(root.data > key){
            root = delete(root.left,key);
        } else if (root.data < key) {
            root = delete(root.right,key);
        }else {
            // we found the key by search in both left subtree and right subtree
            // case 1 if current node have no left child and no right child return null
            if(root.left == null && root.right == null){
                return null;
            }
            // case 2 if left node null
            if(root.left == null){
                return root.right;
                // case 3 if right node null
            } else if (root.right == null) {
                return root.left;
            }else{
                // case 4 if left and right both not null both has child to root ke right main search kro
             TreeRoot  rightRoot = minElement(root.right);
                 root.data =  rightRoot.data;
                 root = delete(rightRoot.right,rightRoot.data);
            }
        }

        return root;
    }


    public TreeRoot minElement(TreeRoot root){
        while(root.left != null){
            root = root.left;
        }
        return root;
    }

    public int getHeight(TreeRoot root){
        int height = 0 ;  int leftHeight = 0 ; int rightHeight = 0;
        if(root==null) return 0;


            TreeRoot l = root.left;
            leftHeight = getHeight(l);
            TreeRoot r = root.right;
            rightHeight = getHeight(r);
        height = Math.max(leftHeight,rightHeight);
       return height+1;
    }


    public static void main(String[] arg){

        // add data in tree
        BinaryTree tree = new BinaryTree();
        tree.add(20);
        tree.add(40);
        tree.add(10);
        tree.add(50);
        // traverse tree inorder
        tree.inorder(root);
        System.out.println(tree.getHeight(root));
    }

}
